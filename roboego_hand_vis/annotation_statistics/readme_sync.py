"""Update existing readme naming fields without regenerating media metadata."""
import json
import copy


def prepare_readme(annotation_path, cleaner, rules, field):
    path = annotation_path.parent.parent / "meta" / "readme.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError(f"readme root must be an object: {path}")
    cleaned = clean_readme_field(document, cleaner, rules, field)
    return (path, cleaned) if cleaned != document else None


def clean_readme_field(document, cleaner, rules, field):
    """Transform one readme field in memory without reading or writing files."""
    document = copy.deepcopy(document)
    if field == "actions":
        adapter = {"actions": {"vocabulary": document.get("actions", [])}}
        document["actions"] = cleaner(adapter, rules)["actions"]["vocabulary"]
    elif field == "objects":
        document["objects"] = cleaner({"objects": document.get("objects", [])}, rules)["objects"]
    elif field == "scene":
        scene = document.get("scene") or {}
        if not isinstance(scene, dict):
            raise ValueError(f"readme.scene must be an object: {path}")
        adapter = {"scene_classification": {"category": scene.get("category"), "area": scene.get("name")}}
        classification = cleaner(adapter, rules)["scene_classification"]
        scene.update(category=classification["category"], name=classification["area"])
        document["scene"] = scene
    else:
        raise ValueError(f"Unsupported readme field: {field}")
    return document


def write_readme(prepared):
    if prepared is not None:
        path, document = prepared
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1
    return 0
