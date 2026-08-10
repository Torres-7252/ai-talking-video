# Gesture Motion and Animated Captions Design

## Goal

Extend the local talking-video pipeline with subtle upper-body and hand motion,
then improve captions with reusable animated styles. The result must continue to
run locally on the current RTX 4060 Laptop GPU (8 GB VRAM) and 16 GB system RAM.

The first acceptance target is a 10-second, 1920x1080 talking video generated
from `avatar/avatar-3d-anime.png` and the existing cloned voice profile.

## Non-goals

- Large or dance-like body motion.
- Training or fine-tuning a motion model.
- Replacing the current Ditto and classic avatar paths.
- Building a timeline editor or general-purpose subtitle editor.
- Rendering a rigged 3D avatar through Blender.

## Selected Approach

Use a sequential hybrid pipeline:

1. Generate cloned speech with the existing GPT-SoVITS provider.
2. Drive the reference image with a rights-cleared, seated-presenter motion clip
   using Tencent MimicMotion.
3. Apply lip synchronization to the generated motion video with the existing
   MuseTalk provider.
4. Generate timed captions with FunASR and the known transcript.
5. Serialize an animated ASS track and burn it in with FFmpeg/libass.
6. Encode the final landscape H.264/AAC output with the current renderer.

MimicMotion was selected because its 16-frame U-Net path can fit in 8 GB VRAM
when VAE decoding is moved to CPU. EchoMimic V2/V3 and HunyuanVideo-Avatar have
higher documented VRAM requirements. PantoMatrix produces 3D motion parameters
rather than directly animating the current flat image.

## Components

### MimicMotion Runtime

Install MimicMotion in an isolated `.venv-mimicmotion` environment and keep its
repository and model weights under
`app/backend/providers/motion/MimicMotion/`. Add a PowerShell installer that is
idempotent and validates the required checkpoints after downloading them.

The motion provider exposes one project-level function:

```python
generate_gesture_motion(
    image_path: Path,
    driver_video_path: Path,
    output_path: Path,
    duration: float,
    intensity: float,
) -> Path
```

The provider preprocesses the driver with DWPose, aligns the pose to the source
subject, and runs landscape inference. Default inference settings are:

- Canvas: 1024x576
- Output rate: 15 fps
- Tile size: 16 frames
- Tile overlap: 6 frames
- Inference steps: 25
- Decode chunk: 1 frame on CPU
- Noise augmentation: 0
- Guidance scale: 2.0

The provider must run a short probe before a full render. The acceptance workflow
uses 3 seconds; the application may cache a successful probe by image, driver,
and settings signature.

### Driver Profiles

Store motion drivers under the ignored `motion/drivers/` directory. The initial
profile is `subtle_presenter`: a seated, front-facing presenter with small
shoulder shifts and one restrained hand-emphasis gesture. Only extracted pose
data influences generation; the driver's appearance and audio are discarded.

Driver motion must remain compatible with the source composition: seated torso,
forearms visible, hands near the desk, and no large reach beyond the original
frame. This reduces hand deformation and background regeneration.

### Pipeline Integration

Add `gesture` as a motion mode without changing the existing `natural` and
`static` modes. For `gesture`, the motion video becomes the MuseTalk input. The
provider environments run sequentially and release CUDA memory between stages.

The pipeline records the motion engine, driver profile, intensity, probe status,
and artifact signature in project metadata. Resume logic invalidates body motion
when the image, driver, duration, or motion settings change.

### Animated Caption Model

Keep the current subtitle JSON as the canonical interchange format. Extend word
timing so every caption segment contains non-overlapping word or character spans.
Use FunASR word timing when it remains compatible with the known transcript;
otherwise distribute the transcript across the recognized segment duration by
visible character weight. Punctuation receives no independent highlight event.

Generate animated ASS directly with standard libass override tags. Do not add
Pycaps, Remotion, or a browser renderer to the production pipeline. This keeps
rendering deterministic and reuses the installed FFmpeg/libass stack.

Provide three caption presets:

- `clean` (default): stable phrase, yellow active word, brief glow, soft phrase
  fade and upward entrance.
- `pop`: short phrase groups, larger type, active group scale pulse, cyan/yellow
  emphasis.
- `bar`: dark translucent bar with a cyan leading rule, highlighted keywords,
  and a restrained slide entrance.

All presets use Microsoft YaHei with a configurable fallback, safe horizontal
margins, maximum two lines, ASS escaping, and a bottom safe area that does not
cover the presenter's hands. Animation must honor segment boundaries and never
extend beyond the audio duration.

### UI and Configuration

Add a `gesture` option to the classic-engine motion selector and show a
driver-profile menu only for that mode. Add a caption-style selector with
`clean`, `pop`, and `bar`. Keep Ditto as the application default so existing
workflows do not unexpectedly invoke a slower diffusion model. The acceptance
render explicitly selects `classic` plus `gesture`, the `subtle_presenter`
driver, motion intensity `0.25`, and caption style `clean`.

The API validates all enum values and clamps neither motion intensity nor timing;
invalid inputs return a 400 response with an actionable message.

## Failure Handling

- Missing runtime or weights: report the exact installer command and preserve
  the existing avatar modes.
- GPU out of memory: retry once at 448-pixel resolution with the same 16-frame
  tile and CPU decode chunk of 1; if it still fails, stop without overwriting a
  previous valid artifact.
- Driver pose mismatch: reject the probe when required torso or arm landmarks
  are missing in more than 20 percent of frames.
- Invalid motion output: require a video stream, positive duration, even frame
  dimensions, and a duration within 0.25 seconds of the requested length.
- Invalid caption timing: reject negative, reversed, or overlapping word spans;
  fall back to deterministic transcript timing only when segment timing is valid.
- Lip-sync failure: retain the generated body-motion video for inspection and
  leave the task resumable from the lip-sync stage.

## Testing

Follow test-first development for each behavior.

- Unit tests for transcript-to-word timing, punctuation handling, ASS escaping,
  preset styles, animation bounds, and line wrapping.
- Unit tests for MimicMotion command/environment construction, signatures,
  fallback settings, and output validation.
- Pipeline tests for stage ordering, resume invalidation, metadata, and CUDA
  environment isolation.
- API tests for motion and caption options plus invalid values.
- A 3-second GPU probe using the anime avatar and selected driver.
- A final 10-second end-to-end render with frame captures near 1.5, 5, and 9
  seconds for visual inspection.

## Acceptance Criteria

- The final file is exactly 10 seconds, 1920x1080, H.264/AAC, and yuv420p.
- Shoulders and torso have visible but restrained motion.
- At least one hand-emphasis gesture is visible without severe extra fingers,
  disconnected limbs, or clothing collapse.
- Lip motion remains synchronized and does not visibly fight the body animation.
- Captions animate in the selected preset, highlight in sync, remain readable,
  and do not cover the face or active hand gesture.
- Existing Ditto and classic generation paths and their tests continue to pass.

## References

- MimicMotion: https://github.com/Tencent/MimicMotion
- libass: https://github.com/libass/libass
- PyonFX inspiration: https://github.com/CoffeeStraw/PyonFX
- EchoMimic V2 comparison: https://github.com/antgroup/echomimic_v2
- PantoMatrix comparison: https://github.com/PantoMatrix/PantoMatrix
