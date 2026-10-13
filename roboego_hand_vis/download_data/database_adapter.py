"""L1 queue orchestration; the downloader does not know about database state."""

from typing import Protocol

from .downloader import DownloadError, KS3Downloader


class TaskService(Protocol):
    def claim(self, table_name: str, data_id: int | None = None) -> dict | None: ...
    def complete(self, table_name: str, data_id: int, local_path: str) -> bool: ...
    def fail(self, table_name: str, data_id: int) -> bool: ...


def download_pending_l1(service: TaskService, downloader: KS3Downloader,
                        limit: int = 0, *, table_name: str = "l1data") -> list[dict]:
    """Claim pending records, download, and write back; zero limit drains the queue.

    A DB writeback error stops work. A successfully published session can be
    reused after an operator recovers its database state. No automatic takeover
    of processing rows or retries of failed rows are performed.
    """
    table_name = table_name.lower()
    if table_name not in ("l1data", "l2data", "l3data"):
        raise ValueError("table must be l1data, l2data, or l3data")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    records = []
    while not limit or len(records) < limit:
        task = service.claim(table_name)
        if task is None:
            break
        record = {"id": task["id"], "ks3_path": task["ks3_path"]}
        try:
            result = downloader.download(task["ks3_path"])
        except DownloadError as exc:
            if not service.fail(table_name, task["id"]):
                raise RuntimeError("Download failed but its database state could not be updated") from exc
            record.update(status="failed", error=str(exc))
        except BaseException:
            # The downloader stops its subprocess before releasing control.
            service.fail(table_name, task["id"])
            raise
        else:
            if not service.complete(table_name, task["id"], result.relative_path):
                raise RuntimeError("Download was published but database completion was rejected")
            record.update(status="completed", local_path=result.relative_path, reused=result.reused)
        records.append(record)
    return records
