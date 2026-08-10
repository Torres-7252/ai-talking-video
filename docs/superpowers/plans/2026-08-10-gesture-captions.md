# Gesture Motion and Animated Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectable animated caption presets and a MimicMotion-based subtle gesture mode, then produce a validated 10-second anime talking video.

**Architecture:** Keep subtitle timing and ASS generation inside the main Python environment, using the existing FFmpeg/libass renderer. Run MimicMotion in an isolated Python 3.10 environment through a small subprocess runner, feed its body-motion video into the existing MuseTalk stage, and preserve Ditto as the default path.

**Tech Stack:** Python 3.10+, `unittest`, FunASR, ASS/libass, FFmpeg, FastAPI, vanilla HTML/JavaScript, MimicMotion commit `6907bdcc259a6a048d41a365e840d22274f9256c`, DWPose, Stable Video Diffusion, MuseTalk.

## Global Constraints

- Target hardware is an RTX 4060 Laptop GPU with 8 GB VRAM and 16 GB RAM.
- Keep `ditto` as the default avatar engine.
- Gesture inference defaults to 1024x576, 15 fps, 16-frame tiles, 6-frame overlap, 25 steps, CPU VAE decoding, and intensity `0.25`.
- Retry one CUDA out-of-memory failure at 448-pixel resolution; never overwrite a previous valid artifact on failure.
- Caption presets are exactly `clean`, `pop`, and `bar`; `clean` is the default.
- Final output is 1920x1080, 25 fps, H.264/AAC, yuv420p, and exactly 10 seconds for the acceptance render.
- Keep third-party repositories, virtual environments, model weights, motion drivers, and generated media ignored by Git.
- Preserve the existing Ditto and LivePortrait/MuseTalk workflows.

## File Structure

- `app/backend/providers/subtitle/__init__.py`: ASR normalization, transcript application, word timing, JSON/SRT orchestration.
- `app/backend/providers/subtitle/ass_renderer.py`: caption presets, ASS escaping, dialogue generation, animation tags.
- `app/backend/providers/motion/mimicmotion.py`: runtime validation, subprocess command construction, OOM fallback, output validation.
- `scripts/mimicmotion_runner.py`: isolated-environment MimicMotion import and inference entry point.
- `scripts/install_mimicmotion_runtime.ps1`: pinned checkout, virtual environment, dependencies, and model download.
- `scripts/pipeline.py`: caption and gesture options, metadata, signatures, stage routing.
- `scripts/web_server.py`: form validation and pipeline argument forwarding.
- `scripts/check_environment.py`: optional MimicMotion readiness reporting.
- `app/frontend/index.html`: gesture and caption selectors with conditional controls.
- `tests/test_subtitle_provider.py`: timing and transcript tests.
- `tests/test_ass_renderer.py`: preset and ASS serialization tests.
- `tests/test_mimicmotion_provider.py`: runtime, command, fallback, and validation tests.
- `tests/test_pipeline.py`: stage ordering, signatures, metadata, and resume tests.
- `tests/test_web_server.py`: API validation and forwarding tests.
- `README.md`: installation and usage documentation.

---

### Task 1: Deterministic Word Timing

**Files:**
- Modify: `app/backend/providers/subtitle/__init__.py`
- Modify: `tests/test_subtitle_provider.py`

**Interfaces:**
- Produces: `allocate_word_timings(text: str, start: float, end: float) -> list[dict]`.
- Produces: `validate_word_timings(words: list[dict], start: float, end: float) -> bool`.
- Updates: every segment returned by `_timed_text_segments()` has timed `words`.

- [ ] **Step 1: Write failing tests for transcript timing**

```python
def test_known_transcript_allocates_non_overlapping_word_timings(self):
    corrected = subtitle.apply_transcript_text(
        [{"text": "old", "start": 0.2, "end": 2.2, "words": []}],
        "\u5927\u5bb6\u597d AI \u8bad\u7ec3\u3002",
    )
    words = [word for segment in corrected for word in segment["words"]]
    self.assertTrue(words)
    self.assertEqual(words[0]["start"], 0.2)
    self.assertEqual(words[-1]["end"], 2.2)
    self.assertTrue(
        all(left["end"] <= right["start"] for left, right in zip(words, words[1:]))
    )

def test_punctuation_does_not_get_an_independent_highlight(self):
    words = subtitle.allocate_word_timings("\u8bad\u7ec3\uff0c\u66f4\u7a33\uff01", 0.0, 1.0)
    self.assertNotIn("\uff0c", [word["text"] for word in words])
    self.assertNotIn("\uff01", [word["text"] for word in words])
    self.assertEqual("".join(word["text"] for word in words), "\u8bad\u7ec3\uff0c\u66f4\u7a33\uff01")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_subtitle_provider.SubtitleProviderTests.test_known_transcript_allocates_non_overlapping_word_timings tests.test_subtitle_provider.SubtitleProviderTests.test_punctuation_does_not_get_an_independent_highlight -v
```

Expected: FAIL because `allocate_word_timings` does not exist and corrected segments have empty `words`.

- [ ] **Step 3: Implement tokenization and weighted timing**

Add an internal tokenizer that groups contiguous ASCII letters/digits, emits CJK characters individually, and appends punctuation to the previous visible token. Allocate duration by visible-character count and force the last token end to the segment end after rounding.

```python
def allocate_word_timings(text: str, start: float, end: float) -> list[dict]:
    if end <= start:
        return []
    tokens = _caption_tokens(_clean_asr_text(text))
    weights = [max(1, len(re.sub(r"[^A-Za-z0-9\u3400-\u9fff]", "", token))) for token in tokens]
    total = sum(weights)
    offset = 0
    words = []
    for token, weight in zip(tokens, weights):
        word_start = start + (end - start) * offset / total
        offset += weight
        word_end = start + (end - start) * offset / total
        words.append({"text": token, "start": round(word_start, 3), "end": round(word_end, 3)})
    words[-1]["end"] = round(end, 3)
    return words
```

Populate each `_timed_text_segments()` segment with timings relative to that segment rather than an empty list. Keep valid FunASR word timing when no known transcript replacement occurs.

- [ ] **Step 4: Run subtitle tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_subtitle_provider -v
```

Expected: all subtitle provider tests PASS.

- [ ] **Step 5: Commit deterministic word timing**

```powershell
git add app/backend/providers/subtitle/__init__.py tests/test_subtitle_provider.py
git commit -m "feat: add deterministic caption word timing"
```

---

### Task 2: Animated ASS Presets

**Files:**
- Create: `app/backend/providers/subtitle/ass_renderer.py`
- Create: `tests/test_ass_renderer.py`
- Modify: `app/backend/providers/subtitle/__init__.py`
- Modify: `app/backend/providers/render/__init__.py`
- Modify: `tests/test_render_provider.py`

**Interfaces:**
- Consumes: timed segments containing `text`, `start`, `end`, and `words`.
- Produces: `CAPTION_PRESETS: frozenset[str]`.
- Produces: `write_animated_ass(subtitles, output_path, preset="clean", width=1920, height=1080) -> Path`.
- Updates: `render_video(..., caption_style: str = "clean") -> Path`.

- [ ] **Step 1: Write failing preset serialization tests**

```python
class AnimatedAssTests(unittest.TestCase):
    def setUp(self):
        self.segment = {
            "text": "\u8bad\u7ec3\uff0c\u66f4\u7a33\u3002",
            "start": 0.0,
            "end": 1.2,
            "words": [
                {"text": "\u8bad", "start": 0.0, "end": 0.3},
                {"text": "\u7ec3\uff0c", "start": 0.3, "end": 0.6},
                {"text": "\u66f4", "start": 0.6, "end": 0.9},
                {"text": "\u7a33\u3002", "start": 0.9, "end": 1.2},
            ],
        }

    def test_clean_preset_emits_base_and_active_word_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "captions.ass"
            ass_renderer.write_animated_ass([self.segment], output, preset="clean")
            content = output.read_text(encoding="utf-8-sig")
        self.assertIn("Style: Clean", content)
        self.assertIn("Dialogue: 0", content)
        self.assertIn("Dialogue: 1", content)
        self.assertIn(r"\fad(", content)
        self.assertIn("&H0047D4FF&", content)

    def test_unknown_preset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "caption preset"):
                ass_renderer.write_animated_ass([], Path(temp_dir) / "x.ass", preset="loud")

    def test_long_caption_wraps_to_at_most_two_lines(self):
        wrapped = ass_renderer.wrap_caption(
            "\u8fd9\u662f\u4e00\u6bb5\u7528\u6765\u9a8c\u8bc1\u4e24\u884c\u5b89\u5168\u5e03\u5c40\u7684\u8f83\u957f\u4e2d\u6587\u5b57\u5e55",
            max_chars=14,
        )
        self.assertLessEqual(wrapped.count(r"\N"), 1)
```

Add render-provider coverage asserting `render_video(..., caption_style="bar")` forwards `bar` to the ASS writer.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m unittest tests.test_ass_renderer tests.test_render_provider -v
```

Expected: FAIL because `ass_renderer` and `caption_style` do not exist.

- [ ] **Step 3: Implement the ASS renderer**

Define immutable preset data for font size, primary color, highlight color, outline, shadow, vertical margin, and entrance tags. Escape braces and newlines before emitting dialogue.

For each segment:

1. Emit one layer-0 base phrase event covering the segment with its preset entrance (`\fad`, `\move`, or both).
2. Emit layer-1 events matching each timed token. Each event redraws the complete phrase while wrapping the active token in highlight tags. `clean` adds a short glow, `pop` applies a 108 percent scale pulse, and `bar` uses the opaque-box ASS border style and leading-rule glyph.
3. Clamp every active event to the segment start/end and skip zero-duration tokens.

Keep `write_ass()` as a compatibility wrapper that calls `write_animated_ass(..., preset="clean")`.

- [ ] **Step 4: Integrate preset selection into rendering**

Change `render_video()` to call:

```python
write_animated_ass(
    subtitles,
    ass_path,
    preset=caption_style,
    width=width,
    height=height,
)
```

No change is required in `build_render_command`; FFmpeg already renders ASS through libass.

- [ ] **Step 5: Run ASS and render tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_ass_renderer tests.test_subtitle_provider tests.test_render_provider -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit animated caption presets**

```powershell
git add app/backend/providers/subtitle app/backend/providers/render/__init__.py tests/test_ass_renderer.py tests/test_subtitle_provider.py tests/test_render_provider.py
git commit -m "feat: add animated ASS caption presets"
```

---

### Task 3: Caption Style in Pipeline, API, and UI

**Files:**
- Modify: `scripts/pipeline.py`
- Modify: `scripts/web_server.py`
- Modify: `app/frontend/index.html`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_web_server.py`

**Interfaces:**
- Produces: `Pipeline(..., caption_style: str = "clean")`.
- Produces: CLI option `--caption-style {clean,pop,bar}`.
- Produces: `/api/generate` form field `caption_style`.

- [ ] **Step 1: Write failing pipeline and API tests**

```python
def test_pipeline_accepts_and_records_caption_style(self):
    parameters = inspect.signature(Pipeline.__init__).parameters
    self.assertIn("caption_style", parameters)
    pipeline = Pipeline("caption-style-test", "Title", "Script", caption_style="bar")
    self.assertEqual(pipeline.caption_style, "bar")
    self.assertEqual(pipeline.metadata["caption_style"], "bar")

def test_unknown_caption_style_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "caption style"):
        Pipeline("bad-caption", "Title", "Script", caption_style="flashy")
```

In `test_web_server.py`, post `caption_style=pop`, patch `Pipeline`, and assert the constructor receives `pop`; post `flashy` and assert HTTP 400.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_pipeline tests.test_web_server -v
```

Expected: FAIL because the new parameter is absent.

- [ ] **Step 3: Implement pipeline and API forwarding**

Validate against `CAPTION_PRESETS`, persist `caption_style` in metadata, pass it to `render_video`, include it in resume metadata, and add the CLI option. Add the FastAPI form field with default `clean` and explicit 400 validation.

- [ ] **Step 4: Add the caption selector to the UI**

Add a compact select beside the template/speed controls:

```html
<div class="field">
  <label for="captionStyle">字幕动效</label>
  <select id="captionStyle">
    <option value="clean" selected>逐词高亮</option>
    <option value="pop">弹跳大字</option>
    <option value="bar">条形重点</option>
  </select>
</div>
```

Append `caption_style` in `doGenerate()` and do not add explanatory help text to the application screen.

- [ ] **Step 5: Run pipeline/API tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_pipeline tests.test_web_server -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit caption controls**

```powershell
git add scripts/pipeline.py scripts/web_server.py app/frontend/index.html tests/test_pipeline.py tests/test_web_server.py
git commit -m "feat: expose animated caption styles"
```

---

### Task 4: MimicMotion Runtime Adapter

**Files:**
- Create: `app/backend/providers/motion/mimicmotion.py`
- Create: `scripts/mimicmotion_runner.py`
- Create: `scripts/install_mimicmotion_runtime.ps1`
- Create: `tests/test_mimicmotion_provider.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `missing_mimicmotion_files(...) -> list[Path]`.
- Produces: `build_mimicmotion_command(image_path, driver_path, output_path, *, duration, intensity, resolution=576) -> tuple[list[str], Path]`.
- Produces: `generate_gesture_motion(image_path: str, driver_video_path: str, output_path: str, duration: float, intensity: float = 0.25) -> Path`.
- Runner CLI consumes `--image`, `--driver`, `--output`, `--duration`, `--intensity`, `--resolution`, `--fps`, `--tile-size`, `--tile-overlap`, `--steps`, and `--decode-chunk-size`.

- [ ] **Step 1: Write failing adapter tests**

```python
class MimicMotionProviderTests(unittest.TestCase):
    def test_command_uses_isolated_runner_and_low_vram_defaults(self):
        command, cwd = mimicmotion.build_mimicmotion_command(
            Path("avatar.png"), Path("driver.mp4"), Path("motion.mp4"),
            duration=10.0, intensity=0.25,
            runtime_root=Path("runtime"), python_executable=Path("python.exe"),
        )
        self.assertEqual(command[0], str(Path("python.exe").resolve()))
        self.assertIn("--tile-size", command)
        self.assertEqual(command[command.index("--tile-size") + 1], "16")
        self.assertEqual(command[command.index("--decode-chunk-size") + 1], "1")
        self.assertEqual(command[command.index("--fps") + 1], "15")
        self.assertEqual(cwd, Path("runtime").resolve())

    def test_cuda_oom_retries_once_at_448(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0, "", "CUDA out of memory" if len(calls) == 1 else "")
        # Patch subprocess.run and output validation, then assert resolutions are 576 and 448.
```

Also test missing runtime messages, driver absence, duration validation, intensity bounds, and atomic replacement of output.

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```powershell
python -m unittest tests.test_mimicmotion_provider -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the host adapter**

Pin these runtime paths:

```python
MIMICMOTION_ROOT = PROJECT_ROOT / "app" / "backend" / "providers" / "motion" / "MimicMotion"
MIMICMOTION_PYTHON = PROJECT_ROOT / ".venv-mimicmotion" / "Scripts" / "python.exe"
MIMICMOTION_RUNNER = PROJECT_ROOT / "scripts" / "mimicmotion_runner.py"
```

Run to a `.tmp.mp4`, capture a `.mimicmotion.log`, retry only when output contains `CUDA out of memory`, validate video/duration, and atomically replace the requested output. Set `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, and `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` in the subprocess environment.

- [ ] **Step 4: Implement the isolated runner**

The runner adds the pinned checkout to `sys.path`, imports MimicMotion inference utilities, extracts/aligned DWPose from the driver, trims or loops pose frames to `ceil(duration * fps)`, loads FP16 model weights, and invokes the pipeline with the exact CLI settings. Move the image encoder and VAE back to CPU after each use, matching the checkout's pipeline behavior; decode one frame at a time. Map UI intensity to pose guidance with `guidance_scale = 1.5 + 2.0 * intensity`, so the default `0.25` becomes the official example value `2.0`.

Use this runner shape rather than invoking the repository CLI through a generated YAML file:

```python
from inference import preprocess
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4

pose_pixels, image_pixels = preprocess(
    args.driver, args.image, resolution=args.resolution, sample_stride=1
)
target_frames = math.ceil(args.duration * args.fps) + 1
pose_pixels = loop_pose_frames(pose_pixels, target_frames)
config = OmegaConf.create({
    "base_model_path": str(runtime / "models" / "stable-video-diffusion-img2vid-xt-1-1"),
    "ckpt_path": str(runtime / "models" / "MimicMotion_1-1.pth"),
})
pipeline = create_pipeline(config, torch.device("cuda"))
frames = pipeline(
    [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5],
    image_pose=pose_pixels,
    num_frames=target_frames,
    tile_size=args.tile_size,
    tile_overlap=args.tile_overlap,
    height=pose_pixels.shape[-2],
    width=pose_pixels.shape[-1],
    fps=args.fps,
    noise_aug_strength=0,
    num_inference_steps=args.steps,
    min_guidance_scale=1.5 + 2.0 * args.intensity,
    max_guidance_scale=1.5 + 2.0 * args.intensity,
    decode_chunk_size=args.decode_chunk_size,
    output_type="pt",
    device=torch.device("cuda"),
).frames.cpu()[0, :, 1:]
save_to_mp4((frames * 255).to(torch.uint8), args.output, fps=args.fps)
```

The runner must fail before inference when fewer than 80 percent of sampled frames have torso and arm landmarks. Emit the ratio in the error message.

- [ ] **Step 5: Implement the pinned PowerShell installer**

Follow `install_liveportrait_runtime.ps1` patterns:

```powershell
$Commit = '6907bdcc259a6a048d41a365e840d22274f9256c'
$Repository = 'https://github.com/Tencent/MimicMotion.git'
$Runtime = Join-Path $ProjectRoot 'app\backend\providers\motion\MimicMotion'
$Venv = Join-Path $ProjectRoot '.venv-mimicmotion'
```

Create Python 3.10 venv, clone and detach at the pinned commit, then install these Windows-compatible pins:

```powershell
python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install diffusers==0.27.0 transformers==4.32.1 huggingface_hub==0.24.7 decord==0.6.0 einops==0.8.1 omegaconf==2.3.0 onnxruntime-gpu==1.18.1 opencv-python==4.10.0.84 matplotlib==3.9.2 tqdm==4.66.5 av==12.3.0
```

Download:

- `tencent/MimicMotion/MimicMotion_1-1.pth`
- `yzd-v/DWPose/yolox_l.onnx`
- `yzd-v/DWPose/dw-ll_ucoco_384.onnx`
- `stabilityai/stable-video-diffusion-img2vid-xt-1-1`

Finish by importing the runner dependencies and checking CUDA availability. Support `-WhatIf` without creating files or making network calls.

- [ ] **Step 6: Update ignored runtime paths**

Add:

```gitignore
app/backend/providers/motion/MimicMotion/
.venv-mimicmotion/
```

- [ ] **Step 7: Run adapter tests and installer dry run**

Run:

```powershell
python -m unittest tests.test_mimicmotion_provider -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_mimicmotion_runtime.ps1 -WhatIf
```

Expected: tests PASS; dry run exits 0 and reports the pinned commit and intended operations.

- [ ] **Step 8: Commit the runtime adapter**

```powershell
git add .gitignore app/backend/providers/motion/mimicmotion.py scripts/mimicmotion_runner.py scripts/install_mimicmotion_runtime.ps1 tests/test_mimicmotion_provider.py
git commit -m "feat: add low-vram MimicMotion adapter"
```

---

### Task 5: Gesture Mode in Pipeline, API, and UI

**Files:**
- Modify: `app/backend/providers/motion/__init__.py`
- Modify: `scripts/pipeline.py`
- Modify: `scripts/web_server.py`
- Modify: `scripts/check_environment.py`
- Modify: `app/frontend/index.html`
- Modify: `tests/test_motion_provider.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `generate_gesture_motion(...)` from Task 4.
- Produces: motion mode `gesture` and driver profile `subtle_presenter`.
- Produces: CLI/API field `driver_profile` with default `subtle_presenter`.
- Produces: `build_motion_signature(..., driver_path: Path | None = None)` that hashes driver bytes for gesture mode.

- [ ] **Step 1: Write failing gesture-routing tests**

```python
def test_gesture_mode_uses_motion_video_for_musetalk(self):
    pipeline = Pipeline(
        "gesture-test", "Title", "Script",
        avatar_engine="classic", motion_mode="gesture",
        driver_profile="subtle_presenter",
    )
    self.assertEqual(pipeline.lipsync_input, pipeline.motion_file)
    self.assertEqual(pipeline.metadata["driver_profile"], "subtle_presenter")

def test_gesture_signature_changes_with_driver_bytes(self):
    first = build_motion_signature(avatar, style="steady", intensity=.25, fps=15, audio_duration=10, driver_path=driver)
    driver.write_bytes(b"changed")
    second = build_motion_signature(avatar, style="steady", intensity=.25, fps=15, audio_duration=10, driver_path=driver)
    self.assertNotEqual(first, second)
```

Add API tests for `motion_mode=gesture`, `driver_profile=subtle_presenter`, and rejection of unknown drivers. Add an environment-check test that missing MimicMotion remains an optional warning while Ditto is available.

- [ ] **Step 2: Run gesture integration tests and verify RED**

Run:

```powershell
python -m unittest tests.test_motion_provider tests.test_pipeline tests.test_web_server -v
```

Expected: FAIL because `gesture` and `driver_profile` are unsupported.

- [ ] **Step 3: Implement pipeline routing and signatures**

Resolve the driver through the strict map `{"subtle_presenter": PROJECT_ROOT / "motion" / "drivers" / "subtle_presenter.mp4"}`. In `step2_motion`, route `natural` to `generate_motion` and `gesture` to `generate_gesture_motion` with audio duration. Update motion labels and validation so both modes require a valid motion artifact before MuseTalk.

Keep `ditto` behavior unchanged: its motion stage is skipped regardless of the selected classic-only motion control.

- [ ] **Step 4: Implement API and UI controls**

Replace the binary LivePortrait toggle with a motion-mode select visible only for the classic engine:

```html
<select id="motionMode" onchange="updateMotionControls()">
  <option value="natural">自然微动</option>
  <option value="gesture">手势动作</option>
  <option value="off">关闭动作</option>
</select>
```

Show the driver select only for `gesture`, keep a stable layout height, and submit both values. Update step labels to display `MimicMotion 手势动作` when selected.

- [ ] **Step 5: Extend environment reporting**

Report required MimicMotion files and the installer command. Mark it as optional unless the request selects gesture mode; do not make the base environment check fail when Ditto remains usable.

- [ ] **Step 6: Run integration tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_motion_provider tests.test_pipeline tests.test_web_server -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit gesture integration**

```powershell
git add app/backend/providers/motion/__init__.py scripts/pipeline.py scripts/web_server.py scripts/check_environment.py app/frontend/index.html tests/test_motion_provider.py tests/test_pipeline.py tests/test_web_server.py
git commit -m "feat: expose MimicMotion gesture mode"
```

---

### Task 6: Runtime Installation and Driver Preparation

**Files:**
- Create locally, ignored: `motion/drivers/subtle_presenter.mp4`
- Create locally, ignored: `motion/drivers/subtle_presenter.source.json`
- Modify: `README.md`

**Interfaces:**
- Produces: a validated 10-second driver at 15 fps with visible torso and arm landmarks.
- Consumes: Pexels source `https://www.pexels.com/video/a-man-talking-and-doing-hand-gestures-8550366/`.

- [ ] **Step 1: Install the pinned MimicMotion runtime**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_mimicmotion_runtime.ps1
```

Expected: exit 0, CUDA import succeeds, and `missing_mimicmotion_files()` returns an empty list.

- [ ] **Step 2: Prepare the restrained driver clip**

Download the selected Pexels clip through its normal free-download flow to `motion/drivers/downloads/pexels-8550366.mp4`. Use the first 10 seconds, remove audio, normalize to 15 fps, and crop/pad to landscape without stretching:

```powershell
$SourceFile = 'motion\drivers\downloads\pexels-8550366.mp4'
$SelectedStart = 0
ffmpeg -hide_banner -loglevel error -ss $SelectedStart -i $SourceFile -t 10 -an -vf "fps=15,scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:(ow-iw)/2:(oh-ih)/2" -c:v libx264 -pix_fmt yuv420p -y motion/drivers/subtle_presenter.mp4
```

Inspect the resulting first section. If it lacks one small emphasis gesture or contains crossed arms, occluded hands, or a large torso turn, reject this source instead of silently changing the fixed profile. Record the source URL, creator `Kindel Media`, Pexels identifier `8550366`, download date, selected start `0`, and the text `Free to use` in `subtle_presenter.source.json`.

- [ ] **Step 3: Validate driver media and landmarks**

Run `ffprobe` to confirm 1024x576, 15 fps, no audio, and 10 seconds. Run the runner's pose-only validation mode and require at least 80 percent valid frames.

- [ ] **Step 4: Document installation and controls**

Add MimicMotion installer, driver placement, CLI examples, caption presets, OOM fallback, and the fact that driver media remains local and ignored.

- [ ] **Step 5: Verify docs and commit**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

```powershell
git add README.md
git commit -m "docs: add gesture motion setup"
```

---

### Task 7: Three-Second Probe and Ten-Second Acceptance Video

**Files:**
- Generated, ignored: `outputs/anime-gesture-probe/*`
- Generated, ignored: `outputs/anime-gesture-10s/*`

**Interfaces:**
- Consumes: `avatar/avatar-3d-anime.png`, existing `default` voice profile, `subtle_presenter` driver, `clean` captions.
- Produces: `outputs/anime-gesture-10s/final.mp4`.

- [ ] **Step 1: Run the complete automated suite before GPU inference**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app scripts tests
git diff --check
```

Expected: zero failures, compile exit 0, and no whitespace errors.

- [ ] **Step 2: Generate and inspect a three-second motion probe**

Use `generate_gesture_motion()` directly with the anime image, first three seconds of the driver, duration 3.0, and intensity 0.25. Extract frames at 0.5, 1.5, and 2.5 seconds. Reject the probe for extra fingers, detached hands, clothing collapse, large background movement, or identity drift.

- [ ] **Step 3: Tune only within the approved bounds if needed**

If the probe fails, first reduce intensity to 0.18. If hands still deform, select a quieter 10-second section from the same rights-cleared source. Do not increase intensity above 0.25 or substitute a large-motion driver.

- [ ] **Step 4: Generate the full body-motion and lip-sync stages**

Generate 10 seconds from `avatar/avatar-3d-anime.png`, then pass the motion video and existing cloned speech to MuseTalk. Keep body and lip-sync artifacts for inspection.

- [ ] **Step 5: Generate clean animated captions and render**

Use the known script, `caption_style=clean`, 1920x1080, and 25 fps. Pad the last video frame and audio silence only when necessary to reach exactly 10.000 seconds.

- [ ] **Step 6: Verify media properties and key frames**

Run:

```powershell
ffprobe -v error -show_entries format=duration -show_entries stream=codec_name,codec_type,width,height,r_frame_rate,pix_fmt,sample_rate,channels -of json outputs/anime-gesture-10s/final.mp4
```

Expected: duration `10.000000`, H.264 video at 1920x1080/25 fps/yuv420p, and AAC audio.

Extract and inspect frames at 1.5, 5, and 9 seconds. Confirm the active caption highlight changes between frames and does not cover the face or moving hands.

- [ ] **Step 7: Run final regression verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app scripts tests
git status --short
```

Expected: zero test failures, compile exit 0, and only the pre-existing untracked anime image plus ignored local assets/outputs remain outside commits.

---

### Task 8: Final Review and Handoff

**Files:**
- Review all committed changes since `a723e26`.

**Interfaces:**
- Produces: verified local server controls and a playable final video path.

- [ ] **Step 1: Review committed scope**

Run:

```powershell
git log --oneline a723e26..HEAD
git diff --stat a723e26..HEAD
git diff --check a723e26..HEAD
```

Expected: only subtitle, gesture runtime/integration, tests, UI controls, and documentation changes.

- [ ] **Step 2: Confirm the local server remains reachable**

Start or reuse the application server and request `http://127.0.0.1:8080/api/environment`. Confirm the page exposes Ditto, classic natural motion, gesture motion, and all three caption presets without console errors.

- [ ] **Step 3: Report evidence**

Provide the final MP4 path, exact duration and stream properties, selected driver/caption settings, test count, any remaining visual limitations, and the local application URL.
