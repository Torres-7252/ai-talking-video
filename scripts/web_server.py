#!/usr/bin/env python3
"""
Web 控制台后端 - FastAPI + WebSocket 实时进度
"""
import asyncio
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app" / "backend" / "providers"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.providers.subtitle.ass_renderer import CAPTION_PRESETS
from scripts.media_library import add_asset, delete_asset, get_asset, list_assets, rename_asset
from scripts.output_paths import OUTPUTS_ROOT
from scripts.long_form_pipeline import LongFormPipeline, segment_limit
from scripts.output_dimensions import dimensions_for_avatar
from scripts.script_segments import split_script_into_segments

# Fix GPT-SoVITS import paths
sys.path.insert(0, str(PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS" / "GPT_SoVITS" / "eres2net"))
sys.path.insert(0, str(PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS" / "GPT_SoVITS"))
sys.path.insert(0, str(PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS"))

app = FastAPI(title="AI数字人口播视频生成器", version="0.2.0")

tasks: dict = {}
ws_clients: list[WebSocket] = []
gpu_job_lock = threading.Lock()
MAX_DITTO_SCRIPT_CHARACTERS = 160


def spoken_character_count(script: str) -> int:
    """Count visible script characters, ignoring formatting whitespace."""
    return len(re.sub(r"\s+", "", script))


def _form_default(value):
    """Use FastAPI Form defaults when the route is called directly in tests."""
    return value.default if hasattr(value, "default") else value


def configured_voice_ids() -> set[str]:
    config_path = PROJECT_ROOT / "config" / "profiles.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        str(profile["id"])
        for profile in config.get("voices", [])
        if profile.get("id")
    }


def _safe_path_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def resolve_project_dir(project_name: str) -> Path:
    safe_name = _safe_path_component(project_name, "project name")
    for outputs in ((OUTPUTS_ROOT / ".work").resolve(), OUTPUTS_ROOT.resolve()):
        project = (outputs / safe_name).resolve()
        if project != outputs and project.is_relative_to(outputs) and project.exists():
            return project
    return ((OUTPUTS_ROOT / ".work").resolve() / safe_name).resolve()


def resolve_project_file(project_name: str, filename: str) -> Path:
    project = resolve_project_dir(project_name)
    target = (project / _safe_path_component(filename, "filename")).resolve()
    if target.parent != project:
        raise ValueError(f"File path escapes project: {filename!r}")
    return target


def get_projects():
    projects = []
    seen = set()
    for outputs_dir in (OUTPUTS_ROOT / ".work", OUTPUTS_ROOT):
        if not outputs_dir.exists():
            continue
        for d in sorted(outputs_dir.iterdir(), reverse=True):
            if d.is_dir() and not d.name.startswith("."):
                if d.name in seen:
                    continue
                meta_file = d / "metadata.json"
                meta = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                final_filename = str(meta.get("final_filename") or "final.mp4")
                final_path = Path(str(meta.get("final_path") or d / final_filename)).resolve()
                projects.append({
                    "name": d.name,
                    "title": meta.get("title", d.name),
                    "created": meta.get("created_at", ""),
                    "duration": meta.get("duration_seconds", 0),
                    "file_size_mb": meta.get("file_size_mb", 0),
                    "steps": meta.get("steps", {}),
                    "has_video": final_path.is_file() or (d / final_filename).exists() or (d / "final.mp4").exists(),
                    "final_filename": final_filename,
                    "has_audio": (d / "audio.wav").exists(),
                    "has_talking": (d / "talking.mp4").exists(),
                    "has_subtitle": (d / "subtitle.json").exists(),
                    "has_packaged": (d / "packaged.mp4").exists(),
                })
                seen.add(d.name)
    for task in tasks.values():
        if task.get("status") != "running":
            continue
        project_name = str(task.get("project_name") or "")
        if not project_name:
            continue
        active_project = {
            "name": project_name,
            "title": task.get("title", project_name),
            "created": task.get("created_at", task.get("id", "")),
            "duration": 0,
            "file_size_mb": 0,
            "steps": task.get("steps", []),
            "status": task.get("status", "running"),
            "segment_count": task.get("segment_count", 1),
            "current_segment": task.get("current_segment", 0),
            "completed_segments": task.get("completed_segments", 0),
            "has_video": False,
            "final_filename": "",
            "has_audio": False,
            "has_talking": False,
            "has_subtitle": False,
            "has_packaged": False,
        }
        if project_name in seen:
            for project in projects:
                if project["name"] == project_name:
                    project.update(active_project)
                    break
        else:
            projects.append(active_project)
            seen.add(project_name)
    return sorted(projects, key=lambda project: project.get("created", ""), reverse=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = PROJECT_ROOT / "app" / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return """<html><body><h1>AI数字人口播视频生成器</h1></body></html>"""


@app.get("/api/projects")
async def api_projects():
    return JSONResponse(get_projects())


@app.post("/api/outputs/open")
async def api_open_outputs():
    """Open the configured output directory in the local file manager."""
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer.exe", str(OUTPUTS_ROOT)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(OUTPUTS_ROOT)])
        else:
            subprocess.Popen(["xdg-open", str(OUTPUTS_ROOT)])
    except OSError as exc:
        return JSONResponse({"error": f"无法打开输出目录: {exc}"}, status_code=500)
    return JSONResponse({"ok": True, "path": str(OUTPUTS_ROOT)})


@app.get("/api/projects/{project_name}")
async def api_project_detail(project_name: str):
    try:
        proj_dir = resolve_project_dir(project_name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not proj_dir.exists():
        return JSONResponse({"error": "项目不存在"}, status_code=404)

    meta_file = proj_dir / "metadata.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    files = {}
    for f in proj_dir.iterdir():
        if f.is_file():
            files[f.name] = {
                "size": f.stat().st_size,
                "url": f"/api/projects/{project_name}/files/{f.name}",
            }
    final_path = Path(str(meta.get("final_path") or "")).resolve() if meta.get("final_path") else None
    final_filename = str(meta.get("final_filename") or "")
    if final_path and final_filename and final_path.is_file():
        files[final_filename] = {
            "size": final_path.stat().st_size,
            "url": f"/api/projects/{project_name}/final",
        }

    return JSONResponse({"name": project_name, "metadata": meta, "files": files})


@app.get("/api/projects/{project_name}/files/{filename}")
async def api_project_file(project_name: str, filename: str):
    try:
        file_path = resolve_project_file(project_name, filename)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not file_path.exists():
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    return FileResponse(str(file_path))


@app.get("/api/projects/{project_name}/final")
async def api_project_final(project_name: str):
    try:
        project = resolve_project_dir(project_name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    metadata_path = project / "metadata.json"
    if not metadata_path.is_file():
        return JSONResponse({"error": "Project does not exist"}, status_code=404)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final_path = Path(str(metadata.get("final_path") or "")).resolve()
    root = OUTPUTS_ROOT.resolve()
    if not final_path.is_file() or not final_path.is_relative_to(root):
        return JSONResponse({"error": "Final video does not exist"}, status_code=404)
    return FileResponse(str(final_path), filename=final_path.name)


@app.delete("/api/projects/{project_name}")
async def api_delete_project(project_name: str):
    """删除项目目录"""
    try:
        proj_dir = resolve_project_dir(project_name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    active_task = next(
        (
            task
            for task in tasks.values()
            if task.get("project_name") == project_name
            and task.get("status") == "running"
        ),
        None,
    )
    if active_task is not None:
        return JSONResponse(
            {"error": "项目正在生成，完成或失败后才能删除"},
            status_code=409,
        )
    if not proj_dir.exists():
        return JSONResponse({"error": "项目不存在"}, status_code=404)

    import shutil
    try:
        shutil.rmtree(str(proj_dir))
        return JSONResponse({"ok": True, "deleted": project_name})
    except Exception as e:
        return JSONResponse({"error": f"删除失败: {str(e)}"}, status_code=500)


async def broadcast(data: dict):
    stale = []
    for ws in ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            stale.append(ws)
    for s in stale:
        ws_clients.remove(s)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.remove(ws)


@app.post("/api/generate")
async def api_generate(
    title: str = Form(""),
    script: str = Form(""),
    voice: str = Form("default"),
    speed: float = Form(1.0),
    template: str = Form("talking_head"),
    resume: bool = Form(False),
    avatar_engine: str = Form("ditto"),
    motion_mode: str = Form("natural"),
    motion_style: str = Form("steady"),
    motion_intensity: float = Form(0.35),
    caption_style: str = Form("clean"),
    driver_profile: str = Form("subtle_presenter"),
    avatar_asset_id: str = Form("default-avatar"),
    voice_asset_id: str = Form(""),
):
    voice = _form_default(voice)
    speed = float(_form_default(speed))
    template = _form_default(template)
    resume = bool(_form_default(resume))
    avatar_engine = _form_default(avatar_engine)
    motion_mode = _form_default(motion_mode)
    motion_style = _form_default(motion_style)
    motion_intensity = float(_form_default(motion_intensity))
    caption_style = _form_default(caption_style)
    driver_profile = _form_default(driver_profile)
    avatar_asset_id = _form_default(avatar_asset_id)
    voice_asset_id = _form_default(voice_asset_id)
    if not isinstance(voice, str):
        voice = "default"
    if not isinstance(avatar_asset_id, str):
        avatar_asset_id = "default-avatar"
    if not isinstance(voice_asset_id, str):
        voice_asset_id = ""
    if not title or not script:
        return JSONResponse({"error": "标题和文案不能为空"}, status_code=400)
    segments = split_script_into_segments(script, segment_limit(speed))
    if voice not in configured_voice_ids():
        return JSONResponse({"error": f"Invalid voice: {voice}"}, status_code=400)
    if avatar_engine != "ditto":
        return JSONResponse(
            {"error": "Only Ditto real mode is available."}, status_code=400
        )
    if caption_style not in CAPTION_PRESETS:
        return JSONResponse(
            {"error": f"Invalid caption style: {caption_style}"}, status_code=400
        )
    if driver_profile not in {"subtle_presenter"}:
        return JSONResponse(
            {"error": f"Invalid driver profile: {driver_profile}"}, status_code=400
        )
    try:
        avatar_asset = get_asset("avatar", avatar_asset_id or "default-avatar", PROJECT_ROOT)
        voice_asset = get_asset("voice", voice_asset_id, PROJECT_ROOT) if voice_asset_id else None
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    from PIL import Image
    try:
        with Image.open(avatar_asset["file_path"]) as avatar_image:
            output_width, output_height = dimensions_for_avatar(*avatar_image.size)
    except OSError:
        # Compatibility fallback for legacy/default assets that have not been created yet.
        output_width, output_height = 1920, 1080

    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(
        c for c in title[:30]
        if c.isascii() and (c.isalnum() or c in " ._-")
    ).strip()
    safe = "_".join(safe.split())
    project_name = f"{task_id}_{safe}" if safe else task_id

    tasks[task_id] = {
        "id": task_id,
        "project_name": project_name,
        "title": title,
        "voice": voice,
        "avatar_asset_id": avatar_asset["id"],
        "voice_asset_id": voice_asset["id"] if voice_asset else "",
        "avatar_engine": "ditto",
        "output_width": output_width,
        "output_height": output_height,
        "caption_style": caption_style,
        "driver_profile": driver_profile,
        "segments": segments,
        "segment_count": len(segments),
        "current_segment": 0,
        "completed_segments": 0,
        "status": "running",
        "current_step": "初始化",
        "steps": [
            {"name": "声音生成", "status": "pending", "icon": "voice"},
            {"name": "Ditto 真实数字人", "status": "pending", "icon": "lipsync"},
            {"name": "字幕生成", "status": "pending", "icon": "subtitle"},
            {"name": "自适应比例合成", "status": "pending", "icon": "render"},
            {"name": "最终导出", "status": "pending", "icon": "export"},
        ],
        "log": [],
        "error": None,
    }
    loop = asyncio.get_event_loop()

    def update_step(name: str, status: str):
        for s in tasks[task_id]["steps"]:
            if s["name"] == name:
                s["status"] = status
                break
        tasks[task_id]["current_step"] = name
        asyncio.run_coroutine_threadsafe(broadcast({"type": "task_update", "task": tasks[task_id]}), loop)

    def add_log(msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[PIPELINE {timestamp}] {msg}")
        tasks[task_id]["log"].append(f"[{timestamp}] {msg}")
        asyncio.run_coroutine_threadsafe(broadcast({"type": "task_update", "task": tasks[task_id]}), loop)

    def update_long_progress(stage: str, status: str, current: int, total: int):
        tasks[task_id]["current_segment"] = current
        tasks[task_id]["segment_count"] = total
        if status == "done" and stage == "最终导出":
            tasks[task_id]["completed_segments"] = current
        update_step(stage, status)

    def execute_pipeline():
        try:
            from scripts.pipeline import Pipeline

            add_log(f"开始生成: {title}")
            update_step("声音生成", "running")

            pipeline_options = {
                "voice_profile": voice,
                "speed": speed,
                "template": template,
                "resume": resume,
                "avatar_engine": "ditto",
                "motion_mode": "off",
                "motion_style": "steady",
                "motion_intensity": 0.0,
                "caption_style": caption_style,
                "driver_profile": "subtle_presenter",
                "output_width": output_width,
                "output_height": output_height,
                "avatar_path": str(avatar_asset["file_path"]),
                "voice_reference_path": str(voice_asset["file_path"]) if voice_asset else None,
                "voice_reference_text": str(voice_asset.get("reference_text") or "") if voice_asset else None,
                "avatar_asset_name": str(avatar_asset.get("name") or ""),
                "voice_asset_name": str(voice_asset.get("name") or "") if voice_asset else "",
            }

            if len(segments) > 1:
                add_log(f"文案已按标点拆分为 {len(segments)} 段，将依次生成后合并")
                final_file = LongFormPipeline(
                    project_name=project_name,
                    title=title,
                    script_text=script,
                    log=add_log,
                    progress=update_long_progress,
                    **pipeline_options,
                ).run()
                for step in tasks[task_id]["steps"]:
                    step["status"] = "done"
                tasks[task_id]["current_step"] = "最终导出"
                tasks[task_id]["status"] = "completed"
                add_log(f"{len(segments)} 段视频已合并完成: {final_file}")
                return

            pipeline = Pipeline(
                project_name=project_name,
                title=title,
                script_text=script,
                **pipeline_options,
            )

            pipeline.step0_setup()
            add_log("文案已保存")

            try:
                pipeline.step1_voice()
                update_step("声音生成", "done")
                add_log("AI声音生成完成")
            except Exception as e:
                update_step("声音生成", "failed")
                add_log(f"声音生成失败: {e}")
                raise

            try:
                update_step("Ditto 真实数字人", "running")
                pipeline.step2_lipsync()
                update_step("Ditto 真实数字人", "done")
                add_log("Ditto 真实数字人生成完成")
            except Exception as e:
                update_step("Ditto 真实数字人", "failed")
                add_log(f"Ditto 真实数字人生成失败: {e}")
                raise

            try:
                update_step("字幕生成", "running")
                pipeline.step3_subtitle()
                update_step("字幕生成", "done")
                add_log("字幕时间轴生成完成")
            except Exception as e:
                update_step("字幕生成", "failed")
                add_log(f"字幕生成失败: {e}")
                raise

            try:
                update_step("自适应比例合成", "running")
                pipeline.step4_render()
                update_step("自适应比例合成", "done")
                add_log(f"视频合成完成: {output_width}x{output_height}")
            except Exception as e:
                update_step("自适应比例合成", "failed")
                add_log(f"视频包装失败: {e}")
                raise

            try:
                update_step("最终导出", "running")
                pipeline.step5_export()
                update_step("最终导出", "done")
                add_log(f"最终视频导出完成: {pipeline.final_file}")
            except Exception as e:
                update_step("最终导出", "failed")
                add_log(f"导出失败: {e}")
                raise

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["project_name"] = project_name
            add_log(f"全部完成！文件: {pipeline.final_file}")

        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
            add_log(f"流水线中断: {e}")

        asyncio.run_coroutine_threadsafe(broadcast({"type": "task_complete", "task": tasks[task_id]}), loop)

    def run_pipeline():
        add_log("等待 GPU 生成队列")
        with gpu_job_lock:
            add_log("已获得 GPU，开始执行生成任务")
            execute_pipeline()

    threading.Thread(target=run_pipeline, daemon=True).start()

    return JSONResponse({"task_id": task_id, "project_name": project_name, "status": "started"})


@app.get("/api/tasks/{task_id}")
async def api_task_status(task_id: str):
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return JSONResponse(tasks[task_id])


@app.get("/api/check")
async def api_check_environment():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_environment.py")],
        capture_output=True, text=True, timeout=60
    )
    return JSONResponse({"output": result.stdout, "returncode": result.returncode})


@app.get("/api/env")
async def api_env_summary():
    """轻量环境摘要"""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        data = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": cuda_ok,
            "gpu": torch.cuda.get_device_name(0) if cuda_ok else "N/A",
            "vram": torch.cuda.get_device_properties(0).total_memory // 1024**2 if cuda_ok else 0,
        }
    except Exception as e:
        data = {
            "python": sys.version.split()[0],
            "torch": "error",
            "cuda": False,
            "gpu": "N/A",
            "vram": 0,
            "error": str(e),
        }
    return JSONResponse(data)


# ═══ 素材上传 ═══

@app.get("/api/assets")
async def api_assets_status():
    """检查素材状态"""
    avatar_image = PROJECT_ROOT / "avatar" / "avatar.jpg"
    ref_audio = PROJECT_ROOT / "voice" / "references" / "default.wav"

    def get_size(p):
        return p.stat().st_size if p.exists() else 0

    def fmt_size(b):
        if b == 0: return "missing"
        if b < 1024: return f"{b}B"
        if b < 1024*1024: return f"{b/1024:.1f}KB"
        return f"{b/(1024*1024):.1f}MB"

    return JSONResponse({
        "avatar": {
            "path": str(avatar_image),
            "exists": avatar_image.exists(),
            "size": fmt_size(get_size(avatar_image)),
            "thumbnail_exists": avatar_image.exists(),
        },
        "voice": {
            "path": str(ref_audio),
            "exists": ref_audio.exists(),
            "size": fmt_size(get_size(ref_audio)),
        },
        "football_assets": {
            "base_path": str(PROJECT_ROOT / "assets" / "football"),
            "folders": [d.name for d in (PROJECT_ROOT / "assets" / "football").iterdir() if d.is_dir()] if (PROJECT_ROOT / "assets" / "football").exists() else [],
        },
        "library": list_assets(PROJECT_ROOT),
    })


@app.get("/api/materials/{kind}/{asset_id}/file")
async def api_material_file(kind: str, asset_id: str):
    try:
        asset = get_asset(kind, asset_id, PROJECT_ROOT)
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return FileResponse(str(asset["file_path"]))


@app.patch("/api/materials/{kind}/{asset_id}")
async def api_rename_material(kind: str, asset_id: str, name: str = Form("")):
    try:
        asset = rename_asset(kind, asset_id, name, PROJECT_ROOT)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse({"ok": True, "asset": {key: value for key, value in asset.items() if key != "file_path"}})


@app.delete("/api/materials/{kind}/{asset_id}")
async def api_delete_material(kind: str, asset_id: str):
    try:
        asset = delete_asset(kind, asset_id, PROJECT_ROOT)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse({"ok": True, "asset": {key: value for key, value in asset.items() if key != "file_path"}})


@app.post("/api/materials/avatar")
async def api_upload_material_avatar(
    file: UploadFile = File(...),
    name: str = Form(""),
):
    if not file.filename:
        return JSONResponse({"error": "No file selected"}, status_code=400)
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        return JSONResponse({"error": "Avatar must be JPG, PNG, or WebP"}, status_code=400)
    content = await file.read()
    if not 1024 <= len(content) <= 50 * 1024 * 1024:
        return JSONResponse({"error": "Avatar file must be between 1KB and 50MB"}, status_code=400)
    destination_dir = PROJECT_ROOT / "materials" / "avatars"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid.uuid4().hex}.jpg"
    try:
        from io import BytesIO
        from PIL import Image

        with Image.open(BytesIO(content)) as image:
            image.load()
            if image.width < 512 or image.height < 512:
                return JSONResponse({"error": "Avatar image must be at least 512px on both sides"}, status_code=400)
            image.convert("RGB").save(destination, "JPEG", quality=95, optimize=True)
        asset = add_asset("avatar", name or Path(file.filename).stem, destination, project_root=PROJECT_ROOT)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        return JSONResponse({"error": f"Could not save avatar: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, "asset": {key: value for key, value in asset.items() if key != "file_path"}})


@app.post("/api/materials/voice")
async def api_upload_material_voice(
    file: UploadFile = File(...),
    name: str = Form(""),
    reference_text: str = Form(""),
):
    if not file.filename:
        return JSONResponse({"error": "No file selected"}, status_code=400)
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "wav"
    allowed = {"wav", "mp3", "flac", "ogg", "m4a", "aac", "wma", "webm", "opus", "weba", "mp4"}
    if ext not in allowed:
        return JSONResponse({"error": "Unsupported audio format"}, status_code=400)
    content = await file.read()
    if not 512 <= len(content) <= 50 * 1024 * 1024:
        return JSONResponse({"error": "Audio file must be between 512B and 50MB"}, status_code=400)
    reference_text = reference_text.strip()
    if not reference_text:
        return JSONResponse({"error": "Reference transcript is required"}, status_code=400)
    destination_dir = PROJECT_ROOT / "materials" / "voices"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid.uuid4().hex}.wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", "pipe:0", "-t", "10", "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le", str(destination)],
            input=content,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            return JSONResponse({"error": f"Audio conversion failed: {detail}"}, status_code=400)
        from app.backend.providers.media_utils import validate_audio

        # Upload must not depend on ASR finding an exact transcript match.  The
        # transcript is still kept for synthesis, while validation ensures the
        # converted recording is usable.
        validate_audio(destination)
        asset = add_asset("voice", name or Path(file.filename).stem, destination, reference_text=reference_text, project_root=PROJECT_ROOT)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        return JSONResponse({"error": f"Could not save voice: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, "asset": {key: value for key, value in asset.items() if key != "file_path"}})


@app.post("/api/assets/avatar")
async def api_upload_avatar(file: UploadFile = File(...)):
    """Upload and normalize a still avatar image."""
    if not file.filename:
        return JSONResponse({"error": "未选择文件"}, status_code=400)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp"):
        return JSONResponse({"error": f"不支持格式: .{ext}，请上传 JPG 或 PNG 图片"}, status_code=400)

    avatar_dir = PROJECT_ROOT / "avatar"
    avatar_dir.mkdir(parents=True, exist_ok=True)

    target = avatar_dir / "avatar.jpg"
    content = await file.read()

    if len(content) < 1024:
        return JSONResponse({"error": "文件太小，请上传有效人物图片"}, status_code=400)
    if len(content) > 50 * 1024 * 1024:
        return JSONResponse({"error": "文件过大，文件大小需小于 50MB"}, status_code=400)

    temporary_target = None
    try:
        from io import BytesIO
        from PIL import Image

        with tempfile.NamedTemporaryFile(
            prefix=".avatar-", suffix=".jpg", dir=avatar_dir, delete=False
        ) as temporary_file:
            temporary_target = Path(temporary_file.name)
        with Image.open(BytesIO(content)) as image:
            image.load()
            if image.width < 512 or image.height < 512:
                return JSONResponse({"error": "图片分辨率过低，宽高需至少 512 像素"}, status_code=400)
            image.convert("RGB").save(
                temporary_target, "JPEG", quality=95, optimize=True
            )
        temporary_target.replace(target)
    except Exception as exc:
        return JSONResponse({"error": f"无法读取人物图片: {exc}"}, status_code=400)
    finally:
        if temporary_target is not None:
            temporary_target.unlink(missing_ok=True)

    return JSONResponse({
        "ok": True,
        "filename": file.filename,
        "saved_as": str(target.name),
        "size": round(target.stat().st_size / (1024 * 1024), 1),
    })


@app.post("/api/assets/voice")
async def api_upload_voice(
    file: UploadFile = File(...),
    reference_text: str = Form("大家好，我是AI足球教练。今天我们来聊一个很多球友都关心的问题。"),
):
    """上传参考音频"""
    try:
        if not file.filename:
            return JSONResponse({"error": "未选择文件"}, status_code=400)

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "wav"
        allowed = ("wav", "mp3", "flac", "ogg", "m4a", "aac", "wma", "webm", "opus", "weba", "mp4")
        if ext not in allowed:
            return JSONResponse({"error": f"不支持格式: .{ext}，支持的格式: wav, mp3, m4a, flac, ogg"}, status_code=400)

        voice_dir = PROJECT_ROOT / "voice" / "references"
        voice_dir.mkdir(parents=True, exist_ok=True)

        content = await file.read()

        if len(content) < 512:
            return JSONResponse({"error": "文件太小，请上传有效音频"}, status_code=400)
        if len(content) > 50 * 1024 * 1024:
            return JSONResponse({"error": "文件过大，文件大小≤50MB"}, status_code=400)

        reference_text = reference_text.strip()
        if not reference_text:
            return JSONResponse({"error": "请填写参考音频对应的原文"}, status_code=400)

        target = voice_dir / "default.wav"
        metadata_target = voice_dir / "default.json"
        temporary_audio = None
        temporary_metadata = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".default-", suffix=".wav", dir=voice_dir, delete=False
            ) as temporary_file:
                temporary_audio = Path(temporary_file.name)
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-i", "pipe:0",
                    "-t", "10", "-ac", "1", "-ar", "32000",
                    "-c:a", "pcm_s16le", str(temporary_audio),
                ],
                input=content,
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                return JSONResponse({"error": f"参考音频转换失败: {detail}"}, status_code=400)

            from app.backend.providers.media_utils import validate_audio

            # ASR alignment is optional quality refinement, not an upload gate.
            # A valid recording with a user-provided transcript must remain usable.
            validate_audio(temporary_audio)
            with tempfile.NamedTemporaryFile(
                prefix=".default-", suffix=".json", dir=voice_dir,
                delete=False, mode="w", encoding="utf-8",
            ) as metadata_file:
                temporary_metadata = Path(metadata_file.name)
                json.dump(
                    {"reference_text": reference_text, "language": "zh"},
                    metadata_file,
                    ensure_ascii=False,
                    indent=2,
                )
            temporary_audio.replace(target)
            temporary_metadata.replace(metadata_target)
        finally:
            if temporary_audio is not None:
                temporary_audio.unlink(missing_ok=True)
            if temporary_metadata is not None:
                temporary_metadata.unlink(missing_ok=True)

        return JSONResponse({
            "ok": True,
            "filename": file.filename,
            "saved_as": str(target.name),
            "size": round(target.stat().st_size / (1024 * 1024), 1),
        })
    except Exception as e:
        return JSONResponse({"error": f"上传失败: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import os

    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8080"))

    print(f"\n{'='*60}")
    print(f"  AI数字人口播视频生成器 v0.2.0")
    print(f"  控制台: http://{host}:{port}")
    print(f"{'='*60}\n")

    # auto-open browser after server starts
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
