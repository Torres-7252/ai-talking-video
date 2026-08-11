#!/usr/bin/env python3
"""
Web 控制台后端 - FastAPI + WebSocket 实时进度
"""
import asyncio
import json
import subprocess
import sys
import tempfile
import threading
import time
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

# Fix GPT-SoVITS import paths
sys.path.insert(0, str(PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS" / "GPT_SoVITS" / "eres2net"))
sys.path.insert(0, str(PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS" / "GPT_SoVITS"))
sys.path.insert(0, str(PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS"))

app = FastAPI(title="AI数字人口播视频生成器", version="0.2.0")

tasks: dict = {}
ws_clients: list[WebSocket] = []


def _safe_path_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def resolve_project_dir(project_name: str) -> Path:
    outputs = (PROJECT_ROOT / "outputs").resolve()
    project = (outputs / _safe_path_component(project_name, "project name")).resolve()
    if project == outputs or not project.is_relative_to(outputs):
        raise ValueError(f"Project path escapes outputs: {project_name!r}")
    return project


def resolve_project_file(project_name: str, filename: str) -> Path:
    project = resolve_project_dir(project_name)
    target = (project / _safe_path_component(filename, "filename")).resolve()
    if target.parent != project:
        raise ValueError(f"File path escapes project: {filename!r}")
    return target


def get_projects():
    outputs_dir = PROJECT_ROOT / "outputs"
    projects = []
    if outputs_dir.exists():
        for d in sorted(outputs_dir.iterdir(), reverse=True):
            if d.is_dir():
                meta_file = d / "metadata.json"
                meta = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                projects.append({
                    "name": d.name,
                    "title": meta.get("title", d.name),
                    "created": meta.get("created_at", ""),
                    "duration": meta.get("duration_seconds", 0),
                    "file_size_mb": meta.get("file_size_mb", 0),
                    "steps": meta.get("steps", {}),
                    "has_video": (d / "final.mp4").exists(),
                    "has_audio": (d / "audio.wav").exists(),
                    "has_talking": (d / "talking.mp4").exists(),
                    "has_subtitle": (d / "subtitle.json").exists(),
                    "has_packaged": (d / "packaged.mp4").exists(),
                })
    return projects


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = PROJECT_ROOT / "app" / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return """<html><body><h1>AI数字人口播视频生成器</h1></body></html>"""


@app.get("/api/projects")
async def api_projects():
    return JSONResponse(get_projects())


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


@app.delete("/api/projects/{project_name}")
async def api_delete_project(project_name: str):
    """删除项目目录"""
    try:
        proj_dir = resolve_project_dir(project_name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
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
):
    if not title or not script:
        return JSONResponse({"error": "标题和文案不能为空"}, status_code=400)
    if avatar_engine not in {"ditto", "classic"}:
        return JSONResponse(
            {"error": f"Invalid avatar engine: {avatar_engine}"}, status_code=400
        )
    if motion_mode not in {"natural", "gesture", "off"}:
        return JSONResponse({"error": f"Invalid motion mode: {motion_mode}"}, status_code=400)
    if motion_style != "steady":
        return JSONResponse({"error": f"Invalid motion style: {motion_style}"}, status_code=400)
    if not 0.0 <= motion_intensity <= 1.0:
        return JSONResponse(
            {"error": "Motion intensity must be between 0.0 and 1.0"},
            status_code=400,
        )
    if caption_style not in CAPTION_PRESETS:
        return JSONResponse(
            {"error": f"Invalid caption style: {caption_style}"}, status_code=400
        )
    if driver_profile not in {"subtle_presenter"}:
        return JSONResponse(
            {"error": f"Invalid driver profile: {driver_profile}"}, status_code=400
        )

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
        "avatar_engine": avatar_engine,
        "motion_mode": motion_mode,
        "motion_style": motion_style,
        "motion_intensity": motion_intensity,
        "caption_style": caption_style,
        "driver_profile": driver_profile,
        "status": "running",
        "current_step": "初始化",
        "steps": [
            {"name": "声音生成", "status": "pending", "icon": "voice"},
            {"name": "LivePortrait 自然动作", "status": "pending", "icon": "motion"},
            {"name": "MuseTalk 1.5口型", "status": "pending", "icon": "lipsync"},
            {"name": "字幕生成", "status": "pending", "icon": "subtitle"},
            {"name": "1080p横屏合成", "status": "pending", "icon": "render"},
            {"name": "最终导出", "status": "pending", "icon": "export"},
        ],
        "log": [],
        "error": None,
    }
    legacy_motion_step_name = tasks[task_id]["steps"][1]["name"]
    legacy_avatar_step_name = tasks[task_id]["steps"][2]["name"]
    motion_step_name = (
        "MimicMotion 手势动作"
        if motion_mode == "gesture"
        else "LivePortrait 自然动作"
    )
    avatar_step_name = (
        "Ditto 真实数字人"
        if avatar_engine == "ditto"
        else "MuseTalk 1.5口型"
    )
    tasks[task_id]["steps"][1]["name"] = motion_step_name
    tasks[task_id]["steps"][2]["name"] = avatar_step_name

    loop = asyncio.get_event_loop()

    def update_step(name: str, status: str):
        name = {
            legacy_motion_step_name: motion_step_name,
            legacy_avatar_step_name: avatar_step_name,
        }.get(name, name)
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

    def run_pipeline():
        try:
            from scripts.pipeline import Pipeline

            add_log(f"开始生成: {title}")
            update_step("声音生成", "running")

            pipeline = Pipeline(
                project_name=project_name,
                title=title,
                script_text=script,
                voice_profile=voice,
                speed=speed,
                template=template,
                resume=resume,
                avatar_engine=avatar_engine,
                motion_mode=motion_mode,
                motion_style=motion_style,
                motion_intensity=motion_intensity,
                caption_style=caption_style,
                driver_profile=driver_profile,
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
                update_step("LivePortrait 自然动作", "running")
                pipeline.step2_motion()
                motion_status = (
                    "done"
                    if avatar_engine == "classic" and motion_mode in {"natural", "gesture"}
                    else "skipped"
                )
                update_step("LivePortrait 自然动作", motion_status)
                if avatar_engine == "ditto":
                    add_log("面部动作将由 Ditto 与口型联合生成")
                elif motion_mode == "natural":
                    add_log("LivePortrait 自然动作生成完成")
                elif motion_mode == "gesture":
                    add_log("MimicMotion 手势动作生成完成")
                else:
                    add_log("自然动作已关闭，使用快速口型模式")
            except Exception as e:
                update_step("LivePortrait 自然动作", "failed")
                add_log(f"自然动作生成失败: {e}")
                raise

            try:
                update_step("MuseTalk 1.5口型", "running")
                pipeline.step2_lipsync()
                update_step("MuseTalk 1.5口型", "done")
                add_log("数字人口型同步完成")
            except Exception as e:
                update_step("MuseTalk 1.5口型", "failed")
                add_log(f"口型同步失败: {e}")
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
                update_step("1080p横屏合成", "running")
                pipeline.step4_render()
                update_step("1080p横屏合成", "done")
                add_log("视频包装完成")
            except Exception as e:
                update_step("1080p横屏合成", "failed")
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
        }
    })


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
            from app.backend.providers.voice import align_reference_audio

            validate_audio(temporary_audio)
            align_reference_audio(temporary_audio, reference_text)
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
