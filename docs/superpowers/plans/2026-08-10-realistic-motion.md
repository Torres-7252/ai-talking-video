# Realistic Portrait Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local LivePortrait natural-motion stage before MuseTalk so the existing portrait produces restrained head, eye, blink, and expression motion while preserving accurate Chinese lip sync and the original background.

**Architecture:** A focused motion provider launches a pinned LivePortrait checkout in an isolated Python 3.10 environment, validates its output, and normalizes the silent motion base to the generated audio duration. The existing pipeline selects either that video or the source image as MuseTalk input; CLI and web controls expose the selection without changing downstream subtitle and render contracts.

**Tech Stack:** Python 3.12 main runtime, isolated Python 3.10 LivePortrait runtime, PyTorch/CUDA FP16, LivePortrait commit `9b294b3d0536135442ea73cb01e6cb3ca7029dd3`, MuseTalk 1.5, FastAPI, vanilla HTML/JavaScript, FFmpeg, pytest.

## Global Constraints

- Run on Windows with NVIDIA RTX 4060 Laptop GPU and 8 GB VRAM.
- Keep GPT-SoVITS, LivePortrait, and MuseTalk in sequential GPU stages; no concurrent model residency.
- Keep `natural` and `off` modes; `natural` is the default and `off` preserves the current fast path.
- Use one `steady` motion style with default intensity `0.35` and 25 fps.
- Preserve the source image background and full 1920x1080 landscape composition.
- Final output remains H.264/AAC, yuv420p, BT.709, 25 fps.
- No paid cloud API, full-body generation, model training, or strong face restoration.

## File Map

- Create `app/backend/providers/motion/__init__.py`: LivePortrait runtime contract, command building, output discovery, duration normalization, validation, and generation.
- Create `tests/test_motion_provider.py`: isolated provider unit tests with mocked subprocess and media probes.
- Create `scripts/install_liveportrait_runtime.ps1`: pinned runtime, weights, and steady-template installer.
- Modify `scripts/check_environment.py`: report LivePortrait runtime/template readiness separately from the existing stack.
- Modify `scripts/pipeline.py`: motion settings, artifact, signature, stage, resume behavior, and CLI flags.
- Modify `app/backend/providers/lipsync/__init__.py`: accept a still image or a validated 25 fps video.
- Modify `scripts/web_server.py`: request parameters and natural-motion progress stage.
- Modify `app/frontend/index.html`: natural-motion toggle and intensity slider.
- Modify `tests/test_pipeline.py`, `tests/test_lipsync_provider.py`, and `tests/test_web_server.py`: stage-selection and API coverage.
- Modify `README.md`: installation, modes, commands, and troubleshooting.

---

### Task 1: LivePortrait Motion Provider Contract

**Files:**
- Create: `app/backend/providers/motion/__init__.py`
- Create: `tests/test_motion_provider.py`

**Interfaces:**
- Consumes: source portrait `Path`, generated audio `Path`, `steady.pkl`, FFmpeg/ffprobe, isolated LivePortrait interpreter.
- Produces: `missing_liveportrait_files() -> list[Path]`, `build_liveportrait_command(...) -> tuple[list[str], Path, Path]`, `normalize_motion_duration(...) -> Path`, and `generate_motion(...) -> Path`.

- [ ] **Step 1: Write failing tests for runtime files and official CLI arguments**

```python
def test_build_liveportrait_command_uses_pinned_runtime_and_intensity(tmp_path):
    runtime = tmp_path / "LivePortrait"
    python = tmp_path / ".venv-liveportrait" / "Scripts" / "python.exe"
    source = tmp_path / "avatar.jpg"
    template = tmp_path / "steady.pkl"
    output = tmp_path / "raw"
    command, cwd, expected = build_liveportrait_command(
        source, template, output, intensity=0.35,
        runtime_root=runtime, python_executable=python,
    )
    assert command[:2] == [str(python), str(runtime / "inference.py")]
    assert command[command.index("-s") + 1] == str(source.resolve())
    assert command[command.index("-d") + 1] == str(template.resolve())
    assert command[command.index("--driving-multiplier") + 1] == "0.35"
    assert cwd == runtime
    assert expected.name == "avatar--steady.mp4"
```

- [ ] **Step 2: Run the provider tests and verify import failure**

Run: `python -m pytest tests/test_motion_provider.py -v`

Expected: FAIL because `app.backend.providers.motion` does not exist.

- [ ] **Step 3: Implement paths, required files, command building, and actionable errors**

```python
PROJECT_ROOT = Path(__file__).resolve().parents[4]
LIVEPORTRAIT_ROOT = PROJECT_ROOT / "app" / "backend" / "providers" / "motion" / "LivePortrait"
LIVEPORTRAIT_PYTHON = PROJECT_ROOT / ".venv-liveportrait" / "Scripts" / "python.exe"
STEADY_TEMPLATE = PROJECT_ROOT / "motion" / "templates" / "steady.pkl"

def build_liveportrait_command(source, driving, output_dir, *, intensity=0.35,
                               runtime_root=LIVEPORTRAIT_ROOT,
                               python_executable=LIVEPORTRAIT_PYTHON):
    source, driving, output_dir = map(lambda p: Path(p).resolve(), (source, driving, output_dir))
    command = [str(python_executable), str(runtime_root / "inference.py"),
               "-s", str(source), "-d", str(driving), "-o", str(output_dir),
               "--driving-multiplier", str(intensity), "--source-max-dim", "1920"]
    expected = output_dir / f"{source.stem}--{driving.stem}.mp4"
    return command, runtime_root, expected
```

- [ ] **Step 4: Add duration normalization tests with an xfade command**

Test that a 10-second motion clip normalized to 30 seconds creates four FFmpeg inputs, chains three `xfade` filters with `duration=0.2`, maps the final label, strips audio, and sets `-t 30.0 -r 25`.

- [ ] **Step 5: Implement `normalize_motion_duration` and `generate_motion`**

`generate_motion` must validate `0.0 <= intensity <= 1.0`, audio duration, source/template/runtime files, run LivePortrait with a 30-minute timeout, write `<output>.liveportrait.log`, validate the raw result, normalize it with 0.2-second crossfades, and atomically replace the requested output only after validation.

- [ ] **Step 6: Run the focused tests**

Run: `python -m pytest tests/test_motion_provider.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the provider**

```powershell
git add app/backend/providers/motion/__init__.py tests/test_motion_provider.py
git commit -m "feat: add LivePortrait motion provider"
```

### Task 2: Isolated Runtime Installer and Environment Check

**Files:**
- Create: `scripts/install_liveportrait_runtime.ps1`
- Modify: `scripts/check_environment.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Git, Python 3.10 launcher, Hugging Face network access, official LivePortrait repository.
- Produces: `.venv-liveportrait/Scripts/python.exe`, pinned `LivePortrait/inference.py`, official model weights, `motion/templates/steady.pkl`.

- [ ] **Step 1: Add a failing environment-check test**

Add `tests/test_motion_provider.py::test_missing_liveportrait_files_lists_interpreter_runtime_weights_and_template` using a temporary root and assert every missing path is returned explicitly.

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_motion_provider.py::test_missing_liveportrait_files_lists_interpreter_runtime_weights_and_template -v`

Expected: FAIL until the required-file contract includes interpreter, inference script, checkpoints, insightface models, and `steady.pkl`.

- [ ] **Step 3: Implement the idempotent PowerShell installer**

The script must:

```powershell
$Commit = '9b294b3d0536135442ea73cb01e6cb3ca7029dd3'
py -3.10 -m venv $Venv
& $Python -m pip install --upgrade pip
git clone https://github.com/KlingAIResearch/LivePortrait.git $Runtime
git -C $Runtime checkout --detach $Commit
& $Python -m pip install -r (Join-Path $Runtime 'requirements.txt')
& $Python -m pip install huggingface_hub[cli]
& (Join-Path $Venv 'Scripts\huggingface-cli.exe') download KlingTeam/LivePortrait --local-dir (Join-Path $Runtime 'pretrained_weights')
```

It must reuse an existing checkout only when `git rev-parse HEAD` equals the pinned commit. It copies the official neutral driving example into `motion/drivers/steady.mp4`, runs LivePortrait once to generate its `.pkl`, copies that template to `motion/templates/steady.pkl`, and leaves the original driver local but outside Git.

Start the script with `[CmdletBinding(SupportsShouldProcess = $true)] param()` and wrap clone, environment creation, package installation, model download, and template generation in `$PSCmdlet.ShouldProcess(...)` checks so `-WhatIf` is a real dry run.

The required human-runtime file contract is:

```text
.venv-liveportrait/Scripts/python.exe
app/backend/providers/motion/LivePortrait/inference.py
app/backend/providers/motion/LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth
app/backend/providers/motion/LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth
app/backend/providers/motion/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth
app/backend/providers/motion/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth
app/backend/providers/motion/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth
app/backend/providers/motion/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx
app/backend/providers/motion/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx
motion/templates/steady.pkl
```

- [ ] **Step 4: Extend `check_environment.py`**

Import `missing_liveportrait_files`, append one `LivePortrait natural motion` check, and print `Run scripts\\install_liveportrait_runtime.ps1` when it fails. The checker remains non-destructive and does not install anything.

- [ ] **Step 5: Document setup and isolated-runtime rationale**

Add commands for installation, environment checking, natural/off modes, expected disk/network work, and Windows CUDA troubleshooting to `README.md`.

- [ ] **Step 6: Verify syntax and current environment behavior**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_liveportrait_runtime.ps1 -WhatIf`

Run: `python scripts/check_environment.py`

Expected before actual installation: existing checks stay readable and only the LivePortrait check may fail.

- [ ] **Step 7: Commit installer and checks**

```powershell
git add scripts/install_liveportrait_runtime.ps1 scripts/check_environment.py README.md tests/test_motion_provider.py
git commit -m "feat: install isolated LivePortrait runtime"
```

### Task 3: Pipeline Stage, Signature, and Resume Semantics

**Files:**
- Modify: `scripts/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `generate_motion(avatar_path, audio_path, output_path, style, intensity, fps) -> Path`.
- Produces: `Pipeline.motion_file`, `Pipeline.step2_motion()`, metadata `motion_mode`, `motion_style`, `motion_intensity`, and a validated MuseTalk input selected by `Pipeline.lipsync_input`.

- [ ] **Step 1: Write failing tests for stage selection and metadata**

```python
def test_natural_mode_generates_motion_before_lipsync(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, motion_mode="natural")
    calls = []
    monkeypatch.setattr("app.backend.providers.motion.generate_motion", lambda **kw: calls.append("motion"))
    monkeypatch.setattr("app.backend.providers.lipsync.generate_lipsync", lambda **kw: calls.append(Path(kw["avatar_path"]).name))
    pipeline.step2_motion()
    pipeline.step3_lipsync()
    assert calls == ["motion", "motion.mp4"]
    assert pipeline.metadata["motion_mode"] == "natural"

def test_off_mode_skips_motion_and_uses_avatar(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, motion_mode="off")
    pipeline.step2_motion()
    assert pipeline.lipsync_input.name == "avatar.jpg"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: FAIL because the new constructor fields and stage do not exist.

- [ ] **Step 3: Implement settings, artifacts, signature, and stage renumbering**

Add constructor arguments `motion_mode: str = "natural"`, `motion_style: str = "steady"`, and `motion_intensity: float = 0.35`. Validate mode/style/range, create `motion.mp4`, and compute a SHA-256 signature from source bytes, style, intensity, fps, and rounded audio duration. Store the signature with the motion step metadata and only resume when the artifact is valid and the signature matches.

- [ ] **Step 4: Route MuseTalk through `lipsync_input`**

```python
@property
def lipsync_input(self) -> Path:
    return self.motion_file if self.motion_mode == "natural" else PROJECT_ROOT / "avatar" / "avatar.jpg"
```

Renumber methods to `step2_motion`, `step3_lipsync`, `step4_subtitle`, `step5_render`, and `step6_export`; preserve metadata keys `lipsync`, `subtitle`, `render`, and `export` so old projects remain readable.

- [ ] **Step 5: Add CLI flags and full run ordering**

Add parser choices for `--motion-mode`, `--motion-style`, and a float `--motion-intensity`. `Pipeline.run()` always calls `step2_motion`; off mode records a skipped stage without creating a fake artifact.

- [ ] **Step 6: Run pipeline tests**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 7: Commit pipeline behavior**

```powershell
git add scripts/pipeline.py tests/test_pipeline.py
git commit -m "feat: add natural motion pipeline stage"
```

### Task 4: MuseTalk Video Input Validation

**Files:**
- Modify: `app/backend/providers/lipsync/__init__.py`
- Modify: `tests/test_lipsync_provider.py`

**Interfaces:**
- Consumes: still images or silent 25 fps motion videos.
- Produces: unchanged `generate_lipsync(...) -> Path` API with input-type validation and the same MuseTalk YAML contract.

- [ ] **Step 1: Write failing tests for video input**

Test that `prepare_lipsync_input` leaves a valid 25 fps even-dimension MP4 unchanged, rejects missing video streams, and normalizes odd image dimensions using the existing PNG path.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_lipsync_provider.py -v`

Expected: FAIL because video-aware preparation does not exist.

- [ ] **Step 3: Implement media-aware preparation**

```python
def prepare_lipsync_input(path: Path, output_path: Path) -> Path:
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        info = validate_video(path)
        # Reject no-video, non-25-fps, and odd-dimension inputs with actionable messages.
        return path.resolve()
    return prepare_even_avatar(path, output_path)
```

Keep MuseTalk command options `--fps 25 --batch_size 1 --use_float16` unchanged.

- [ ] **Step 4: Run lipsync tests**

Run: `python -m pytest tests/test_lipsync_provider.py -v`

Expected: PASS.

- [ ] **Step 5: Commit video-input support**

```powershell
git add app/backend/providers/lipsync/__init__.py tests/test_lipsync_provider.py
git commit -m "feat: accept motion video in MuseTalk adapter"
```

### Task 5: Web Controls and Progress Stage

**Files:**
- Modify: `scripts/web_server.py`
- Modify: `app/frontend/index.html`
- Modify: `tests/test_web_server.py`

**Interfaces:**
- Consumes: multipart fields `motion_mode`, `motion_style`, `motion_intensity`.
- Produces: validated `Pipeline` arguments, a visible natural-motion progress row, and persisted user selection.

- [ ] **Step 1: Write failing API tests**

Submit `/api/generate` with `motion_mode=natural`, `motion_style=steady`, and `motion_intensity=0.35`; patch `Pipeline` and assert those exact constructor arguments. Add 400-response tests for an unknown mode and intensity outside `0.0..1.0`.

- [ ] **Step 2: Run web tests and verify failure**

Run: `python -m pytest tests/test_web_server.py -v`

Expected: FAIL because the form fields are not accepted.

- [ ] **Step 3: Implement API validation and progress sequencing**

Add form parameters with the approved defaults. Insert a `LivePortrait 自然动作` task step between voice and MuseTalk, call `pipeline.step2_motion()`, and skip its visual state cleanly when mode is off. Call the renumbered downstream methods.

- [ ] **Step 4: Add compact controls to the existing form**

Add a labeled toggle for natural motion and a stable-width range input for intensity. JavaScript submits `natural` or `off`, submits `steady`, and disables the slider while off. Add the LivePortrait icon mapping and progress element without redesigning the page.

- [ ] **Step 5: Run web tests**

Run: `python -m pytest tests/test_web_server.py -v`

Expected: PASS.

- [ ] **Step 6: Commit web support**

```powershell
git add scripts/web_server.py app/frontend/index.html tests/test_web_server.py
git commit -m "feat: expose natural motion controls"
```

### Task 6: Regression Verification and Runtime Installation

**Files:**
- Modify only when a failing test reveals an in-scope defect.

**Interfaces:**
- Consumes: all changes from Tasks 1-5.
- Produces: passing automated suite and a ready local LivePortrait runtime.

- [ ] **Step 1: Run formatting and complete unit suite**

Run: `git diff --check`

Run: `python -m pytest -v`

Expected: no whitespace errors and all tests pass.

- [ ] **Step 2: Install the pinned runtime and weights**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_liveportrait_runtime.ps1`

Expected: Python 3.10 environment exists, checkout is at `9b294b3d...`, weights are complete, and `motion/templates/steady.pkl` exists.

- [ ] **Step 3: Run the full environment check**

Run: `python scripts/check_environment.py`

Expected: every check, including `LivePortrait natural motion`, reports OK.

- [ ] **Step 4: Commit any installer-generated tracked template metadata only**

Do not commit model weights, virtual environments, cloned third-party repositories, raw driver videos, or output videos. Ensure `.gitignore` covers them; commit only small project-owned configuration or documentation needed to reproduce installation.

### Task 7: Real A/B Acceptance Video

**Files:**
- Outputs: `outputs/realism-natural/final.mp4`
- Outputs: `outputs/realism-fast/final.mp4`
- Outputs: stage logs and `metadata.json` in both directories.

**Interfaces:**
- Consumes: current `avatar/avatar.jpg`, `voice/references/default.wav`, a 20-30 second Chinese acceptance script.
- Produces: comparable natural and off outputs plus ffprobe evidence.

- [ ] **Step 1: Generate the fast-path baseline**

Run the CLI with project `realism-fast`, the acceptance script, `--motion-mode off`, and `--speed 1.0`.

Expected: the existing MuseTalk-only output completes unchanged.

- [ ] **Step 2: Generate the natural-motion version**

Run the same CLI arguments with project `realism-natural`, `--motion-mode natural`, `--motion-style steady`, and `--motion-intensity 0.35`.

Expected: LivePortrait completes before MuseTalk with no CUDA OOM.

- [ ] **Step 3: Probe both videos**

Run `ffprobe` JSON checks for codec, pixel format, dimensions, frame rate, duration, audio stream, color primaries, transfer, and color space.

Expected: 1920x1080, 25 fps, H.264/AAC, yuv420p, BT.709, and audio/video duration difference no greater than 0.1 second.

- [ ] **Step 4: Perform visual acceptance**

Inspect representative frames and both full videos. Accept only when the natural version has restrained blink/eye/head motion, no obvious loop jump, no persistent mouth-edge flicker, and no visible drift in identity, hair, shirt logos, hands, or background.

- [ ] **Step 5: Record final verification**

Update `README.md` with the verified command and output path, run `python -m pytest -v` once more, and commit only source/tests/docs changes.
