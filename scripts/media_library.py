"""Persistent, named media assets for the talking-video factory."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIRNAME = "materials"
MANIFEST_NAME = "library.json"
KINDS = {"avatar", "voice"}


def _library_dir(project_root: Path = PROJECT_ROOT) -> Path:
    return Path(project_root) / LIBRARY_DIRNAME


def _manifest_path(project_root: Path = PROJECT_ROOT) -> Path:
    return _library_dir(project_root) / MANIFEST_NAME


def _relative_path(project_root: Path, path: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(project_root).resolve())).replace("\\", "/")


def _safe_asset_path(project_root: Path, relative_path: str) -> Path:
    root = Path(project_root).resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Asset path escapes the project")
    return candidate


def _read_manifest(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = _manifest_path(project_root)
    if not path.is_file():
        return {"version": 1, "assets": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("assets"), list):
        raise RuntimeError(f"Invalid media library manifest: {path}")
    return data


def _write_manifest(data: dict[str, Any], project_root: Path = PROJECT_ROOT) -> None:
    directory = _library_dir(project_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(project_root)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def ensure_default_assets(project_root: Path = PROJECT_ROOT) -> None:
    """Expose the legacy default files as selectable material-library entries."""
    root = Path(project_root)
    data = _read_manifest(root)
    existing = {str(asset.get("id")) for asset in data["assets"]}
    defaults = (
        ("default-avatar", "avatar", "默认人物", root / "avatar" / "avatar.jpg", ""),
        ("default-voice", "voice", "我的声音", root / "voice" / "references" / "default.wav", ""),
    )
    changed = False
    for asset_id, kind, name, path, reference_text in defaults:
        if asset_id in existing or not path.is_file():
            continue
        data["assets"].append(
            {
                "id": asset_id,
                "kind": kind,
                "name": name,
                "path": _relative_path(root, path),
                "reference_text": reference_text,
                "created_at": datetime.now().isoformat(),
                "system": True,
            }
        )
        changed = True
    if changed:
        _write_manifest(data, root)


def list_assets(project_root: Path = PROJECT_ROOT) -> dict[str, list[dict[str, Any]]]:
    root = Path(project_root)
    ensure_default_assets(root)
    entries: dict[str, list[dict[str, Any]]] = {"avatar": [], "voice": []}
    for asset in _read_manifest(root)["assets"]:
        kind = str(asset.get("kind"))
        if kind not in KINDS:
            continue
        try:
            path = _safe_asset_path(root, str(asset.get("path") or ""))
        except ValueError:
            continue
        if not path.is_file():
            continue
        entries[kind].append(
            {
                "id": str(asset.get("id")),
                "kind": kind,
                "name": str(asset.get("name") or path.stem),
                "filename": path.name,
                "size": path.stat().st_size,
                "reference_text": str(asset.get("reference_text") or ""),
                "created_at": str(asset.get("created_at") or ""),
                "system": bool(asset.get("system")),
            }
        )
    for kind in entries:
        entries[kind].sort(key=lambda asset: (asset["system"], asset["created_at"]), reverse=True)
    return entries


def get_asset(kind: str, asset_id: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"Unsupported asset kind: {kind}")
    if not asset_id or not asset_id.isascii() or not asset_id.replace("-", "").isalnum():
        raise ValueError("Invalid asset id")
    root = Path(project_root)
    ensure_default_assets(root)
    for asset in _read_manifest(root)["assets"]:
        if asset.get("kind") == kind and asset.get("id") == asset_id:
            path = _safe_asset_path(root, str(asset.get("path") or ""))
            if not path.is_file():
                raise FileNotFoundError(f"Material file does not exist: {asset_id}")
            return {**asset, "file_path": path}
    raise FileNotFoundError(f"Material does not exist: {asset_id}")


def add_asset(
    kind: str,
    name: str,
    path: Path,
    *,
    reference_text: str = "",
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"Unsupported asset kind: {kind}")
    clean_name = " ".join(str(name).split())[:80]
    if not clean_name:
        raise ValueError("Material name cannot be empty")
    root = Path(project_root)
    source = Path(path).resolve()
    relative = _relative_path(root, source)
    data = _read_manifest(root)
    asset = {
        "id": uuid.uuid4().hex,
        "kind": kind,
        "name": clean_name,
        "path": relative,
        "reference_text": reference_text.strip() if kind == "voice" else "",
        "created_at": datetime.now().isoformat(),
        "system": False,
    }
    data["assets"].append(asset)
    _write_manifest(data, root)
    return {**asset, "file_path": source}


def rename_asset(kind: str, asset_id: str, name: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    clean_name = " ".join(str(name).split())[:80]
    if not clean_name:
        raise ValueError("Material name cannot be empty")
    root = Path(project_root)
    data = _read_manifest(root)
    for asset in data["assets"]:
        if asset.get("kind") == kind and asset.get("id") == asset_id:
            _safe_asset_path(root, str(asset.get("path") or ""))
            asset["name"] = clean_name
            _write_manifest(data, root)
            return {**asset, "file_path": _safe_asset_path(root, str(asset["path"]))}
    raise FileNotFoundError(f"Material does not exist: {asset_id}")


def delete_asset(kind: str, asset_id: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Remove an uploaded material and its managed file, never a system default."""
    if kind not in KINDS:
        raise ValueError(f"Unsupported asset kind: {kind}")
    if not asset_id or not asset_id.isascii() or not asset_id.replace("-", "").isalnum():
        raise ValueError("Invalid asset id")

    root = Path(project_root)
    ensure_default_assets(root)
    data = _read_manifest(root)
    for index, asset in enumerate(data["assets"]):
        if asset.get("kind") != kind or asset.get("id") != asset_id:
            continue
        if asset.get("system"):
            raise ValueError("System materials cannot be deleted")

        path = _safe_asset_path(root, str(asset.get("path") or ""))
        data["assets"].pop(index)
        _write_manifest(data, root)
        library_dir = _library_dir(root).resolve()
        if path.is_file() and path.is_relative_to(library_dir):
            path.unlink()
        return {**asset, "file_path": path}
    raise FileNotFoundError(f"Material does not exist: {asset_id}")
