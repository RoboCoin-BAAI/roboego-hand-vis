"""Freeze an explicit KS3 MCAP inventory into L1-style batches and path CSVs."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from roboego_hand_vis.upload_data.ks3 import atomic_json, prefix

SOURCE = "ks3://baai-eai-pwm/LightwheelAI/EgoStandard/"
DESTINATION = "ks3://baai-eai-pwm/LightwheelAI/EgoStandard_lerobot/"


def plan(objects, source, destination, batch_size=100):
    source, destination = prefix(source), prefix(destination)
    if batch_size < 1 or destination.startswith(source) or source.startswith(destination):
        raise ValueError("Invalid batch size or overlapping source/upload prefixes")
    selected = sorted((x for x in objects if x["uri"].endswith(".mcap")), key=lambda x:x["uri"])
    if not selected or len({x["uri"] for x in selected}) != len(selected):
        raise ValueError("Empty or duplicate MCAP inventory")
    names = Counter(Path(x["uri"].rsplit("/",1)[-1]).stem for x in selected)
    batches = {}
    for i, obj in enumerate(selected):
        uri = obj["uri"]
        if not uri.startswith(source) or obj["size"] <= 0:
            raise ValueError("Invalid source object: " + uri)
        prefix(uri)  # Validate key without interpreting spaces/# as shell syntax.
        identity = hashlib.sha256(uri.encode()).hexdigest()
        name = Path(uri.rsplit("/",1)[-1]).stem
        if not re.fullmatch(r"[A-Za-z0-9_-]+",name):
            name = "session_"+identity[:24]
        elif names[name] > 1:
            name += "_"+identity[:16]
        batch = f"batch_{i//batch_size+1:07d}"
        batches.setdefault(batch, []).append(dict(id=identity,session_name=name,
            source_uri=uri,size=obj["size"],etag=obj["etag"],
            destination_uri=destination+batch+"/"+name+"/"))
    return dict(source_prefix=source,destination_prefix=destination,batch_size=batch_size,
                total_objects=len(selected),batches=batches)


def write_plan(root, value):
    root = Path(root)
    # Stable batch membership is essential for previously uploaded destinations.
    inventory = root/"inventory.json"
    if inventory.exists() and json.loads(inventory.read_text(encoding="utf-8")) != value:
        raise ValueError("Inventory changed; use a new state-root rather than renumbering existing batches")
    for batch, items in value["batches"].items():
        path = root/batch/"manifest.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != items:
            raise ValueError("Existing batch plan differs: " + batch)
    atomic_json(inventory, value)
    for batch, items in value["batches"].items():
        atomic_json(root/batch/"manifest.json",items)
        buf = io.StringIO(newline="")
        writer = csv.writer(buf)
        writer.writerow(["目标路径"])
        writer.writerows([item["source_uri"]] for item in items)
        # A JSON atomic writer is not appropriate for literal CSV text.
        path = root/(batch+".csv")
        temp = path.with_suffix(".csv.tmp")
        temp.write_text(buf.getvalue(),encoding="utf-8-sig",newline="")
        temp.replace(path)
