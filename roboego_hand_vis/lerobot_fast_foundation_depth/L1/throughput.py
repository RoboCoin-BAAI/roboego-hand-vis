"""Persistent per-GPU workers with dynamically claimed episode tasks."""
from contextlib import redirect_stdout, redirect_stderr
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback


def gpu_worker(device, tasks, results, options, engine_factory=None):
    # This happens before importing Torch or the model code in a fresh process.
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["OMP_NUM_THREADS"] = str(options.get("cpu_threads", 2))
    os.environ["MKL_NUM_THREADS"] = str(options.get("cpu_threads", 2))
    try:
        if engine_factory is None:
            from .engine import DepthEngine
            engine_factory = DepthEngine
        started = time.perf_counter()
        engine = engine_factory(options)
        results.put(dict(kind="ready", gpu=device, pid=os.getpid(), model_load_s=time.perf_counter() - started))
    except BaseException:
        results.put(dict(kind="worker_error", gpu=device, error=traceback.format_exc()))
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        log = None
        try:
            offset = options.get("right_frame_offset", 0)
            suffix = f"_right_offset_{'m' if offset < 0 else 'p'}{abs(offset)}" if offset else ""
            log = Path(task["session"]) / "annotation" / f"depth_{Path(task['episode']).parent.name}_{Path(task['episode']).name}{suffix}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
                try:
                    report = engine.process(task)
                except Exception:
                    traceback.print_exc()
                    raise
            results.put(dict(kind="result", task_id=task["task_id"], passed=True, gpu=device, log=str(log), **report))
        except Exception:
            results.put(dict(kind="result", task_id=task["task_id"], passed=False, gpu=device,
                             log=str(log), error=traceback.format_exc()))


def run_jobs(jobs, devices, options, engine_factory=None):
    if not jobs:
        return []
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    devices = devices[:len(jobs)]
    processes = [ctx.Process(target=gpu_worker, args=(gpu, tasks, results, options, engine_factory)) for gpu in devices]
    reports = {}
    try:
        for process in processes:
            process.start()
        for task in jobs:
            tasks.put(task)
        for _ in processes:
            tasks.put(None)
        while len(reports) < len(jobs):
            try:
                report = results.get(timeout=.5)
            except queue.Empty:
                if all(not p.is_alive() for p in processes):
                    break
                continue
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if report["kind"] == "result":
                reports[report["task_id"]] = report
        for process in processes:
            process.join()
        # A failed model startup or dead CUDA worker must not hang the batch.
        for task in jobs:
            if task["task_id"] not in reports:
                reports[task["task_id"]] = dict(kind="result", task_id=task["task_id"], passed=False,
                    error="GPU workers exited before reporting this task", episode=task["episode"])
                print(json.dumps(reports[task["task_id"]]), flush=True)
        return [reports[task["task_id"]] for task in jobs]
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join()
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
