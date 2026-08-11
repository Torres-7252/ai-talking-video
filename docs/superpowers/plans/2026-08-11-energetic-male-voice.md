# Energetic Male Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local CosyVoice Chinese energetic male preset to the talking-video pipeline and web UI.

**Architecture:** Keep CosyVoice in an isolated Python 3.10 virtual environment and invoke it through a small runner. Route the existing `generate_voice` API by provider profile so all downstream video stages remain unchanged.

**Tech Stack:** Python 3.10, CosyVoice-300M-Instruct, ModelScope, FastAPI, vanilla HTML/JavaScript, unittest.

## Global Constraints

- Preserve the existing `default` GPT-SoVITS profile.
- Pin the official CosyVoice repository to `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`.
- Keep model weights, virtual environments, and generated samples out of Git.
- Use the existing `voice_profile` pipeline contract.

---

### Task 1: Isolated CosyVoice Runtime

**Files:**
- Create: `scripts/install_cosyvoice.py`
- Create: `scripts/cosyvoice_runner.py`
- Modify: `.gitignore`
- Test: `tests/test_cosyvoice_provider.py`

**Interfaces:**
- Runner arguments: `--text`, `--output`, `--speaker`, `--instruct`, `--speed`.
- Runner output: validated mono WAV at the requested absolute path.

- [ ] **Step 1: Write failing tests for pinned runtime paths and runner command construction.**
- [ ] **Step 2: Run `python -m unittest tests.test_cosyvoice_provider -v` and confirm missing interfaces fail.**
- [ ] **Step 3: Implement the installer, isolated runner, and ignored runtime paths.**
- [ ] **Step 4: Install dependencies and `CosyVoice-300M-Instruct`; generate a short Chinese male sample.**
- [ ] **Step 5: Re-run provider tests and commit the runtime integration.**

### Task 2: Backend Voice Routing

**Files:**
- Create: `app/backend/providers/voice/cosyvoice.py`
- Modify: `app/backend/providers/voice/__init__.py`
- Modify: `config/profiles.json`
- Test: `tests/test_voice_provider.py`

**Interfaces:**
- `generate_cosyvoice(text: str, output_path: str, profile: dict, speed: float) -> Path`.
- `generate_voice(...)` selects `gpt-sovits` or `cosyvoice` from the configured profile.

- [ ] **Step 1: Write failing tests for provider lookup and energetic-male dispatch.**
- [ ] **Step 2: Run the focused tests and confirm dispatch is absent.**
- [ ] **Step 3: Add the energetic profile and minimal provider routing.**
- [ ] **Step 4: Run focused tests and verify both old and new profiles pass.**
- [ ] **Step 5: Commit backend routing.**

### Task 3: Web Voice Selection

**Files:**
- Modify: `app/frontend/index.html`
- Modify: `scripts/web_server.py`
- Test: `tests/test_frontend_contract.py`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Form field: `voice` with values `default` and `energetic_male`.
- API rejects voice IDs not present in `config/profiles.json`.

- [ ] **Step 1: Write failing contract tests for the selector and submitted field.**
- [ ] **Step 2: Run focused frontend and API tests and confirm failure.**
- [ ] **Step 3: Add the selector, request field, validation, and task metadata.**
- [ ] **Step 4: Run focused tests and inspect the rendered form at desktop and mobile widths.**
- [ ] **Step 5: Commit web integration.**

### Task 4: End-to-End Verification

**Files:**
- No tracked file changes required.

**Interfaces:**
- Input: energetic-male profile and Chinese promotional script.
- Output: playable WAV and final MP4 with matching audio duration.

- [ ] **Step 1: Generate an energetic male sample and recognize it independently with FunASR.**
- [ ] **Step 2: Run `python -m unittest discover -s tests -v` with zero failures.**
- [ ] **Step 3: Restart the local server and verify `http://127.0.0.1:8080` returns HTTP 200.**
- [ ] **Step 4: Submit a short video task with `voice=energetic_male` and verify final media streams.**
- [ ] **Step 5: Commit any final test fixes and report the sample and server URL.**
