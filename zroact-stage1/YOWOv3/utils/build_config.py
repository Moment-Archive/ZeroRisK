from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_LIKE_KEYS = {
    "config_path",
    "pretrain_path",
    "data_root",
    "save_folder",
    "labelmap",
    "groundtruth",
    "detections",
}


def _resolve_path(value, config_dir):
    if value in (None, ""):
        return value

    path = Path(value)
    if path.is_absolute():
        return str(path)

    candidates = [
        config_dir / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Fall back to the project root so relative outputs land in a stable place.
    return str((PROJECT_ROOT / path).resolve())


def _resolve_config_paths(node, config_dir, parent_key=None):
    if isinstance(node, dict):
        resolved = {}
        for key, value in node.items():
            next_value = _resolve_config_paths(value, config_dir, key)
            if key in PATH_LIKE_KEYS and isinstance(next_value, str):
                next_value = _resolve_path(next_value, config_dir)
            elif parent_key == "PRETRAIN" and isinstance(next_value, str):
                next_value = _resolve_path(next_value, config_dir)
            resolved[key] = next_value
        return resolved

    if isinstance(node, list):
        return [_resolve_config_paths(item, config_dir, parent_key) for item in node]

    return node

def build_config(config_file='config/ucf_config.yaml'):
    config_path = Path(config_file).resolve()

    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    config = _resolve_config_paths(config, config_path.parent)

    if config['active_checker']:
        pass
    
    return config
