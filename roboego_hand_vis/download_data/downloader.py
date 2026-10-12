"""KS3 subprocess and filesystem downloads without database dependencies."""

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from urllib.parse import urlsplit
from uuid import uuid4


RECEIPT = ".ks3_download.json"


class DownloadError(RuntimeError):
    """A session was not downloaded and published successfully."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    relative_path: str
    reused: bool = False


def session_source(source: str) -> tuple[str, str]:
    """Normalize a session prefix, keeping a trailing slash to avoid siblings."""
    parsed = urlsplit(source)
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme != "ks3" or not parsed.netloc or not parsed.path.strip("/")
            or parsed.query or parsed.fragment or source != source.strip()
            or "\\" in source or "\x00" in source
            or any(part in (".", "..", "") for part in parts)
            or parts[-1] in (".downloads", ".locks")):
        raise DownloadError("Expected a ks3://bucket/path/session directory prefix")
    return source.rstrip("/") + "/", parts[-1]


class KS3Downloader:
    """Publish complete sessions under output_root/session_name.

    A receipt verifies the source and file sizes for reuse. Interrupted or failed
    transfers stay in .downloads for inspection, never in the final directory.
    All cooperating processes must use the same output root and locking support.
    """

    def __init__(self, output_root: Path, tool: Path, config: Path, jobs: int = 8):
        self.output_root = Path(output_root).expanduser().resolve()
        self.tool = Path(tool).expanduser().resolve()
        self.config = Path(config).expanduser().resolve()
        if not self.tool.is_file() or not os.access(self.tool, os.X_OK):
            raise DownloadError("KS3 tool must exist and be executable")
        if not self.config.is_file():
            raise DownloadError("KS3 configuration file does not exist")
        if not 1 <= jobs <= 10000:
            raise DownloadError("jobs must be between 1 and 10000")
        self.jobs = jobs
        self.output_root.mkdir(parents=True, exist_ok=True)

    def download(self, source: str) -> DownloadResult:
        source, name = session_source(source)
        locks = self.output_root / ".locks"
        locks.mkdir(exist_ok=True)
        # Prevent overlapping direct/DB workers from publishing the same folder.
        with (locks / (name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._download_locked(source, name)

    def _download_locked(self, source: str, name: str) -> DownloadResult:
        destination = self.output_root / name
        if destination.exists() or destination.is_symlink():
            self._verify_existing(destination, source)
            return DownloadResult(destination, name, reused=True)

        work = self.output_root / ".downloads" / uuid4().hex
        payload = work / "payload"
        reports = work / "reports"
        payload.mkdir(parents=True)
        reports.mkdir()
        log_path = work / "ks3.log"
        command = [
            str(self.tool), "cp", source, str(payload), "-r", "-f",
            "-j", str(self.jobs), "-c", str(self.config),
            "--output-dir", str(reports), "--output-item", "failed",
            "--checkpoint-dir", str(work / "checkpoint"), "--loglevel=off",
        ]
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait()
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    raise
            if code != 0:
                raise DownloadError(f"KS3 exited with code {code}")
            for manifest in reports.glob("*failed*"):
                if not manifest.is_file():
                    continue
                lines = [line for line in manifest.read_text(errors="replace").splitlines() if line.strip()]
                # ks3util manifests may contain only their command header when
                # no file failed. Any error record or extra row means failure.
                if (len(lines) > 1 or any("error message:" in line for line in lines)
                        or (lines and " cp " not in lines[0])):
                    raise DownloadError("KS3 reported failed files in its result manifest")
            files = {}
            for path in payload.rglob("*"):
                if path.is_symlink():
                    raise DownloadError("Downloaded payload contains a symbolic link")
                if path.is_file():
                    files[path.relative_to(payload).as_posix()] = path.stat().st_size
            if not files:
                raise DownloadError("KS3 returned no files for the session prefix")
            if RECEIPT in files:
                raise DownloadError("Remote payload conflicts with the reserved download receipt")
            (payload / RECEIPT).write_text(json.dumps({"source": source, "files": files}, indent=2))
            if destination.exists() or destination.is_symlink():
                raise DownloadError("Destination appeared during download; refusing to overwrite")
            payload.rename(destination)
        except (OSError, DownloadError) as exc:
            raise DownloadError(f"{exc}; temporary data and log: {work}") from exc
        # Publication already succeeded; cleanup failure does not invalidate it.
        shutil.rmtree(work, ignore_errors=True)
        return DownloadResult(destination, name)

    def _verify_existing(self, destination: Path, source: str) -> None:
        try:
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Destination is not a regular directory")
            receipt = json.loads((destination / RECEIPT).read_text())
            if receipt["source"] != source or not receipt["files"]:
                raise ValueError("Source does not match the existing download")
            for relative, size in receipt["files"].items():
                path = destination / relative
                if (path.is_symlink() or not path.resolve().is_relative_to(destination.resolve())
                        or not path.is_file() or path.stat().st_size != size):
                    raise ValueError("A recorded file is missing or changed")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise DownloadError(f"Existing directory cannot be safely reused: {destination}") from exc
