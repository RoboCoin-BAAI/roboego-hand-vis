"""KS3 transfers with explicit object names and post-upload size verification."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from uuid import uuid4


def prefix(uri):
    if not re.match(r"^ks3://[^/]+/.+", uri) or "\\" in uri or any(c in uri for c in ("\0", "\n", "\r")):
        raise ValueError("Expected KS3 prefix: " + uri)
    if any(p in (".", "..", "") for p in uri.split("/", 3)[3].rstrip("/").split("/")):
        raise ValueError("Unsafe KS3 key: " + uri)
    return uri.rstrip("/") + "/"


def parse_listing(text):
    objects = []
    for line in text.splitlines():
        if "ks3://" not in line:
            continue
        before, key = line.split("ks3://", 1)
        columns = before.split()
        if len(columns) < 3 or not columns[-3].isdigit():
            raise ValueError("Unexpected KS3 listing line: " + line[:300])
        objects.append(dict(uri="ks3://"+key.strip(), size=int(columns[-3]), etag=columns[-1].strip('"')))
    return objects


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("."+path.name+"."+uuid4().hex)
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class KS3:
    def __init__(self, tool, config):
        self.tool, self.config = str(tool), str(config)

    def _run(self, arguments):
        result = subprocess.run([self.tool, *map(str, arguments), "-c", self.config,
                                 "--loglevel=off"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"KS3 exited {result.returncode}: " + (result.stdout+result.stderr)[-4000:])
        return result.stdout

    def objects(self, uri):
        return parse_listing(self._run(["ls", uri]))

    def copy(self, source, destination):
        # The outer workflow owns concurrency. Do not create another large pool.
        self._run(["cp", source, destination, "-f", "-j", "1", "--parallel", "1"])

    def download(self, item, target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        def valid(path, transferred=False):
            if not path.is_file() or path.stat().st_size != item["size"]:
                return False
            if re.fullmatch(r"[0-9a-fA-F]{32}", item["etag"]):
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "md5").hexdigest() == item["etag"].lower():
                        return True
            # ETag is an opaque object identifier, including some 32-hex ETags.
            # A fresh successful ks3util transfer verifies CRC64 by default.
            # Existing files without a matching MD5 must be transferred again.
            return transferred
        if valid(target):
            return
        partial = target.with_name(target.name+".part")
        self.copy(item["source_uri"], str(partial))
        if not valid(partial, transferred=True):
            actual = partial.stat().st_size if partial.is_file() else None
            raise RuntimeError(f"Downloaded size mismatch: expected={item['size']}, actual={actual}: " + item["source_uri"])
        partial.replace(target)

    def upload_directory(self, local, destination, *, recursive_copy=False):
        local, destination = Path(local).resolve(), prefix(destination)
        files = {}
        for path in sorted(local.rglob("*")):
            if path.is_symlink():
                raise ValueError("Refusing symlink in upload: " + str(path))
            if path.is_file():
                files[path.relative_to(local).as_posix()] = path.stat().st_size
        if not files or "meta/info.json" not in files:
            raise ValueError("No converted dataset to upload")
        if recursive_copy:
            # Same cp/update/jobs convention as download_data.base. The trailing
            # slash copies the session contents into the exact destination prefix.
            self._run(['-r', '-u', '--read-timeout', '6', '--jobs=64',
                       'cp', str(local) + '/', destination])
        else:
            # Other workflows retain their existing per-object transfer policy.
            for name in files:
                self.copy(str(local/name), destination+name)
        actual = {x["uri"][len(destination):]: x["size"] for x in self.objects(destination)
                  if x["uri"].startswith(destination) and not x["uri"].endswith("/")}
        mismatched = [name for name, size in files.items() if actual.get(name) != size]
        if mismatched:
            raise RuntimeError("Upload verification failed: " + ", ".join(mismatched[:10]))
        return dict(destination_uri=destination, files=len(files), bytes=sum(files.values()),
                    verification="all uploaded object sizes matched", file_sizes=files)
