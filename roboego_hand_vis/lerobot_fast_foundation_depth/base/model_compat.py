"""Explicit compatibility defaults for older serialized stereo checkpoints."""
def patch_model_config(model, checkpoint_config):
    from omegaconf import OmegaConf, open_dict

    args = model.args
    current = args.get("normalize") if hasattr(args, "get") else getattr(args, "normalize", None)
    if current is not None:
        return {}
    # Upstream make_plugin_onnx.py also uses normalize=True when the
    # serialized model omits this field. Prefer an explicit adjacent cfg.yaml.
    value = checkpoint_config.get("normalize", True)
    if not isinstance(value, bool):
        raise ValueError("Checkpoint normalize must be a boolean")
    if OmegaConf.is_config(args):
        with open_dict(args):
            args.normalize = value
    elif isinstance(args, dict):
        args["normalize"] = value
    else:
        args.normalize = value
    checkpoint_config["normalize"] = value
    return dict(normalize=value)
