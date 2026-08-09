# Local AI Talking Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Windows project generate a local 1920x1080 Chinese talking-head video from the approved still image and cloned reference voice.

**Architecture:** Keep the existing staged pipeline, but replace cwd-dependent model loading and mismatched subprocess calls with validated adapters. GPT-SoVITS and MuseTalk 1.5 run sequentially so only one large model occupies the RTX 4060 Laptop GPU at a time; FunASR and FFmpeg produce timed subtitles and the final media file.

**Tech Stack:** Python 3.12, `unittest`, PyTorch 2.5.1+cu124, GPT-SoVITS v3, MuseTalk 1.5 FP16, FunASR 1.4.1, FastAPI, FFmpeg 8.1.2.

## Global Constraints

- Run locally on Windows with an NVIDIA GeForce RTX 4060 Laptop GPU and 8 GB VRAM.
- Use `C:\Users\31078\Desktop\60b37dcd61ff34e82bace8beb9a5b679.jpg` as the complete 16:9 person and background image.
- Use `C:\Users\31078\Desktop\标准录音 1.mp3` as the source recording without modifying the original.
- Use MuseTalk 1.5 with FP16 and 25 fps.
- Output `final.mp4` as 1920x1080 H.264 video, AAC audio, and `yuv420p` pixels.
- Keep the full image composition; do not create a portrait crop, full-body motion, gestures, or camera motion.
- Do not call paid cloud video or speech APIs.

---

### Task 1: Media Validation and Approved Assets

**Files:**
- Create: `app/backend/providers/media_utils.py`
- Create: `scripts/prepare_assets.py`
- Create: `tests/test_media_utils.py`
- Create during execution: `avatar/avatar.jpg`
- Create during execution: `voice/references/default.wav`
- Create during execution: `voice/references/default.json`

**Interfaces:**
- Produces: `probe_media(path: Path) -> dict`, `validate_audio(path: Path) -> dict`, `validate_video(path: Path, expected_size: tuple[int, int] | None = None) -> dict`.
- Produces: `prepare_assets(image_source: Path, voice_source: Path, project_root: Path) -> dict[str, Path]`.

- [ ] **Step 1: Write failing media validation tests**

```python
class MediaValidationTests(unittest.TestCase):
    def test_validate_audio_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            validate_audio(Path("missing.wav"))

    def test_probe_media_parses_ffprobe_json(self):
        completed = subprocess.CompletedProcess([], 0, '{"streams":[{"codec_type":"audio"}],"format":{"duration":"2.5"}}', "")
        with mock.patch("media_utils.subprocess.run", return_value=completed):
            info = probe_media(Path("sample.wav"))
        self.assertEqual(info["duration"], 2.5)
        self.assertEqual(info["streams"][0]["codec_type"], "audio")
```

- [ ] **Step 2: Run the tests and confirm the expected import failure**

Run: `python -m unittest tests.test_media_utils -v`

Expected: FAIL because `media_utils` does not exist.

- [ ] **Step 3: Implement the media helpers and asset preparation**

`probe_media` runs ffprobe with JSON output and normalizes duration. `prepare_assets` copies the JPG and runs this exact audio conversion through a list-form subprocess command:

```text
ffmpeg -y -ss 1.54 -to 7.73 -i <source.mp3> -ac 1 -ar 32000 -c:a pcm_s16le voice/references/default.wav
```

Write `default.json` as UTF-8 JSON containing:

```json
{
  "reference_text": "大家好，我是AI足球教练。今天我们来聊一个很多球友都关心的问题。",
  "language": "zh"
}
```

- [ ] **Step 4: Run unit and real asset checks**

Run: `python -m unittest tests.test_media_utils -v`

Run: `python scripts/prepare_assets.py --image "C:\Users\31078\Desktop\60b37dcd61ff34e82bace8beb9a5b679.jpg" --voice "C:\Users\31078\Desktop\标准录音 1.mp3"`

Run: `ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -show_entries format=duration -of json voice/references/default.wav`

Expected: tests PASS; WAV is mono, 32000 Hz, and approximately 6.19 seconds.

- [ ] **Step 5: Commit the task**

```text
git add app/backend/providers/media_utils.py scripts/prepare_assets.py tests/test_media_utils.py avatar/avatar.jpg voice/references/default.wav voice/references/default.json
git commit -m "feat: prepare validated talking video assets"
```

### Task 2: GPT-SoVITS Absolute Configuration and Voice Clone

**Files:**
- Modify: `app/backend/providers/voice/__init__.py`
- Modify: `config/profiles.json`
- Create: `tests/test_voice_provider.py`

**Interfaces:**
- Produces: `load_voice_profile(name: str) -> dict`.
- Produces: `build_tts_config() -> dict` with absolute v3 model paths.
- Preserves: `generate_voice(text: str, output_path: str, voice_profile: str = "default", speed: float = 1.0, api_url: str | None = None) -> Path`.

- [ ] **Step 1: Write failing voice configuration tests**

```python
class VoiceProviderTests(unittest.TestCase):
    def test_tts_config_paths_are_absolute_and_exist(self):
        config = voice.build_tts_config()["custom"]
        for key in ("t2s_weights_path", "vits_weights_path", "bert_base_path", "cnhuhbert_base_path"):
            self.assertTrue(Path(config[key]).is_absolute())
            self.assertTrue(Path(config[key]).exists(), key)

    def test_profile_uses_recording_transcript(self):
        profile = voice.load_voice_profile("default")
        self.assertEqual(profile["reference_text"], "大家好，我是AI足球教练。今天我们来聊一个很多球友都关心的问题。")
```

- [ ] **Step 2: Run the tests and confirm the missing API failure**

Run: `python -m unittest tests.test_voice_provider -v`

Expected: FAIL because `build_tts_config` and `load_voice_profile` do not exist.

- [ ] **Step 3: Implement absolute GPT-SoVITS configuration**

Build the `TTS_Config` from a dictionary with `custom.version="v3"`, `device="cuda"`, `is_half=True`, and these absolute files:

```text
GPT_SoVITS/pretrained_models/s1v3.ckpt
GPT_SoVITS/pretrained_models/s2Gv3.pth
GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
GPT_SoVITS/pretrained_models/chinese-hubert-base-hf
```

Pass the profile's `reference_text` as `prompt_text`; never derive it from target speech. Validate the generated WAV with `validate_audio`. Keep `unload_voice()` responsible for deleting the singleton, collecting Python objects, and clearing CUDA cache.

- [ ] **Step 4: Verify tests and real short speech generation**

Run: `python -m unittest tests.test_voice_provider -v`

Run: `python scripts/generate_voice.py --text "大家好，这是本地口播测试。" --output outputs/acceptance/audio.wav`

Run: `ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -show_entries format=duration -of json outputs/acceptance/audio.wav`

Expected: tests PASS and a non-silent WAV is generated without cwd-related file errors.

- [ ] **Step 5: Commit the task**

```text
git add app/backend/providers/voice/__init__.py config/profiles.json tests/test_voice_provider.py
git commit -m "fix: load GPT-SoVITS with absolute local paths"
```

### Task 3: MuseTalk 1.5 Models, Preflight, and CLI Adapter

**Files:**
- Create: `scripts/download_musetalk_models.ps1`
- Modify: `scripts/check_environment.py`
- Modify: `app/backend/providers/lipsync/__init__.py`
- Fix: `scripts/generate_lipsync.py`
- Create: `tests/test_lipsync_provider.py`

**Interfaces:**
- Produces: `required_musetalk_files() -> list[Path]`.
- Produces: `build_musetalk_job(avatar_path: Path, audio_path: Path, output_path: Path) -> tuple[dict, list[str], Path]`.
- Preserves: `generate_lipsync(avatar_path: str, audio_path: str, output_path: str, use_fp16: bool = True, avatar_cache_dir: str | None = None) -> Path`.

- [ ] **Step 1: Write failing MuseTalk command tests**

```python
class MuseTalkProviderTests(unittest.TestCase):
    def test_command_matches_v15_cli(self):
        job, command, cwd = lipsync.build_musetalk_job(Path("avatar.jpg"), Path("audio.wav"), Path("talking.mp4"))
        self.assertIn("--version", command)
        self.assertIn("v15", command)
        self.assertIn("--use_float16", command)
        self.assertIn("--inference_config", command)
        self.assertNotIn("--avatar", command)
        self.assertEqual(job["task_0"]["video_path"], str(Path("avatar.jpg").resolve()))
        self.assertEqual(cwd, lipsync.MUSETALK_PATH)
```

- [ ] **Step 2: Run the test and confirm the current contract failure**

Run: `python -m unittest tests.test_lipsync_provider -v`

Expected: FAIL because the provider still emits unsupported `--avatar` and `--fp16` arguments.

- [ ] **Step 3: Implement the resumable model download script**

The PowerShell script verifies free disk space, installs `huggingface_hub[hf_xet]` if `hf` is missing, then resumes official downloads into the existing MuseTalk `models` directory:

```text
hf download TMElyralab/MuseTalk --local-dir models
hf download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae
hf download openai/whisper-tiny --local-dir models/whisper
hf download yzd-v/DWPose --local-dir models/dwpose --include dw-ll_ucoco_384.pth
hf download ManyOtherFunctions/face-parse-bisent --local-dir models/face-parse-bisent
```

The script exits nonzero if any required file is absent or any `.incomplete` file remains.

- [ ] **Step 4: Implement the official v1.5 adapter**

Write a per-project YAML containing `video_path`, `audio_path`, and `result_name`. Execute `python -m scripts.inference` with `cwd=MUSETALK_PATH`, `--version v15`, `--use_float16`, `--batch_size 1`, the v1.5 UNet/config paths, and an isolated result directory. Require both return code zero and a valid generated MP4 before copying to the pipeline output. Remove the obsolete training preprocess call for still images.

- [ ] **Step 5: Download models and run checks**

Run: `powershell -ExecutionPolicy Bypass -File scripts/download_musetalk_models.ps1`

Run: `python scripts/check_environment.py`

Run: `python -m unittest tests.test_lipsync_provider -v`

Expected: no incomplete weights; environment and unit tests pass.

- [ ] **Step 6: Run a real MuseTalk acceptance clip**

Run: `python scripts/generate_lipsync.py --avatar avatar/avatar.jpg --audio outputs/acceptance/audio.wav --output outputs/acceptance/talking.mp4`

Run: `ffprobe -v error -show_entries stream=codec_type,codec_name,width,height -show_entries format=duration -of json outputs/acceptance/talking.mp4`

Expected: MP4 contains video and audio streams and duration matches the speech within 0.5 seconds.

- [ ] **Step 7: Commit the task**

```text
git add scripts/download_musetalk_models.ps1 scripts/check_environment.py app/backend/providers/lipsync/__init__.py scripts/generate_lipsync.py tests/test_lipsync_provider.py
git commit -m "fix: integrate MuseTalk 1.5 local inference"
```

### Task 4: Timed Subtitles and FFmpeg-Only Rendering

**Files:**
- Modify: `app/backend/providers/subtitle/__init__.py`
- Modify: `app/backend/providers/render/__init__.py`
- Create: `tests/test_subtitle_provider.py`
- Create: `tests/test_render_provider.py`

**Interfaces:**
- Produces: `normalize_asr_result(result: list, audio_duration: float) -> list[dict]`.
- Produces: `write_ass(subtitles: list[dict], output_path: Path, width: int = 1920, height: int = 1080) -> Path`.
- Preserves: `render_video(...) -> Path` while implementing it through FFmpeg.

- [ ] **Step 1: Write failing subtitle normalization tests**

```python
class SubtitleTests(unittest.TestCase):
    def test_top_level_funasr_timestamps_become_nonempty_segments(self):
        result = [{"text": "大家好我是AI教练", "timestamp": [[0, 300], [320, 620], [650, 900]]}]
        subtitles = subtitle.normalize_asr_result(result, 1.0)
        self.assertTrue(subtitles)
        self.assertGreater(subtitles[0]["end"], subtitles[0]["start"])

    def test_empty_recognition_is_rejected(self):
        with self.assertRaises(RuntimeError):
            subtitle.normalize_asr_result([], 2.0)
```

- [ ] **Step 2: Write failing render command tests**

```python
class RenderTests(unittest.TestCase):
    def test_render_command_is_landscape_ffmpeg(self):
        command = render.build_render_command(Path("talking.mp4"), Path("subtitle.ass"), Path("final.mp4"))
        joined = " ".join(command)
        self.assertIn("1920:1080", joined)
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertNotIn("moviepy", joined.lower())
```

- [ ] **Step 3: Run tests and confirm missing behavior**

Run: `python -m unittest tests.test_subtitle_provider tests.test_render_provider -v`

Expected: FAIL because normalization and command builders are absent.

- [ ] **Step 4: Implement local ASR normalization and ASS output**

Support both `sentence_info` and top-level `text` plus `timestamp`. Reject empty or invalid timelines. Generate ASS with PlayResX 1920, PlayResY 1080, white text, black outline, centered lower safe-area placement, and UTF-8 encoding.

- [ ] **Step 5: Replace MoviePy rendering with FFmpeg**

Use list-form subprocess arguments, scale and pad without cropping, burn ASS subtitles, encode H.264/AAC at 25 fps, and validate the output through ffprobe.

- [ ] **Step 6: Verify unit and real rendering tests**

Run: `python -m unittest tests.test_subtitle_provider tests.test_render_provider -v`

Run: `python scripts/generate_subtitles.py --audio outputs/acceptance/audio.wav --output outputs/acceptance/subtitle.json`

Run: `python scripts/render_video.py --video outputs/acceptance/talking.mp4 --subtitles outputs/acceptance/subtitle.json --output outputs/acceptance/final.mp4`

Expected: tests PASS and final MP4 is 1920x1080 with visible burned subtitles.

- [ ] **Step 7: Commit the task**

```text
git add app/backend/providers/subtitle/__init__.py app/backend/providers/render/__init__.py tests/test_subtitle_provider.py tests/test_render_provider.py
git commit -m "fix: render timed subtitles with FFmpeg"
```

### Task 5: Validated Resume Pipeline and Web Path Safety

**Files:**
- Modify: `scripts/pipeline.py`
- Modify: `scripts/web_server.py`
- Modify: `app/frontend/index.html`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_web_server.py`

**Interfaces:**
- Produces: `artifact_is_valid(step: str, path: Path) -> bool`.
- Produces: `resolve_project_file(project_name: str, filename: str) -> Path`.
- Pipeline default avatar becomes `avatar/avatar.jpg`; final output remains `outputs/<project>/final.mp4`.

- [ ] **Step 1: Write failing resume validation test**

```python
class PipelineTests(unittest.TestCase):
    def test_resume_does_not_skip_empty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "audio.wav"
            output.touch()
            pipeline = Pipeline("case", "title", "text", resume=True)
            self.assertTrue(pipeline.should_run(output, "voice"))
```

- [ ] **Step 2: Write failing path traversal tests**

```python
class WebPathTests(unittest.TestCase):
    def test_project_file_rejects_parent_escape(self):
        with self.assertRaises(ValueError):
            web_server.resolve_project_file("project", "../../.env")

    def test_project_file_stays_under_outputs(self):
        path = web_server.resolve_project_file("project", "final.mp4")
        self.assertTrue(path.is_relative_to(web_server.PROJECT_ROOT / "outputs"))
```

- [ ] **Step 3: Run tests and confirm current failures**

Run: `python -m unittest tests.test_pipeline tests.test_web_server -v`

Expected: FAIL because resume checks only existence and file routes accept unsanitized path pieces.

- [ ] **Step 4: Implement validated stage state and secure paths**

Every stage records start, finish, status, error, log path, and artifact probe. Resume skips only validated media. Resolve all project file requests and require the resolved target to remain under `outputs`. Update the UI labels to identify MuseTalk 1.5 and the 1920x1080 landscape result without redesigning the page.

- [ ] **Step 5: Run all unit tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS with no unexpected warnings or tracebacks.

- [ ] **Step 6: Commit the task**

```text
git add scripts/pipeline.py scripts/web_server.py app/frontend/index.html tests/test_pipeline.py tests/test_web_server.py
git commit -m "fix: validate resumable web generation jobs"
```

### Task 6: End-to-End Acceptance and Operator Documentation

**Files:**
- Modify: `README.md`
- Modify: `启动控制台.ps1`
- Create during execution: `outputs/acceptance/final.mp4`

**Interfaces:**
- Documents: one supported launch command and one acceptance pipeline command.
- Verifies: final output streams, duration, dimensions, frame rate, and non-static mouth-region frames.

- [ ] **Step 1: Run the complete pipeline with a short script**

Run:

```text
python scripts/pipeline.py --project acceptance --title "本地AI口播测试" --script "大家好，这是本地AI口播视频测试。今天我们一起练好第一脚触球。" --template talking_head
```

Expected: all six metadata stages are `done` and `outputs/acceptance/final.mp4` exists.

- [ ] **Step 2: Probe the final media**

Run:

```text
ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt -show_entries format=duration,size -of json outputs/acceptance/final.mp4
```

Expected: 1920x1080, 25 fps, H.264, AAC, `yuv420p`, nonzero size, and duration within 0.5 seconds of `audio.wav`.

- [ ] **Step 3: Verify visible frame changes around the mouth**

Extract frames at 20%, 50%, and 80% of the clip with FFmpeg and compare a fixed face-region crop using OpenCV. Require at least one pair to have nonzero mean absolute pixel difference; visually inspect the extracted frames for intact identity and background.

- [ ] **Step 4: Verify the web console**

Run: `python scripts/web_server.py`

Open: `http://127.0.0.1:8080`

Submit the same short script and confirm progress, failure messages, video preview, and download route work.

- [ ] **Step 5: Update operating documentation**

Document model download, asset preparation, environment check, launch, expected first-run duration, output location, resume usage, and common CUDA/FFmpeg errors. Keep `启动控制台.ps1` as the supported one-click entry point.

- [ ] **Step 6: Run final verification**

Run: `python -m unittest discover -s tests -v`

Run: `python scripts/check_environment.py`

Run: the ffprobe command from Step 2.

Expected: all tests pass, environment check reports every required item, and final video meets every acceptance property.

- [ ] **Step 7: Commit the task**

```text
git add README.md 启动控制台.ps1 docs/superpowers/plans/2026-08-09-local-ai-talking-video.md
git commit -m "docs: document local talking video workflow"
```
