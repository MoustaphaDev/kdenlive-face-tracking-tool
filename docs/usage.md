# Usage Guide

This guide covers CLI workflows, flags, example commands, and how to use the generated mask effects inside Kdenlive.

Back to [README](../README.md). For install choices, provider/platform notes, diagnostics, and troubleshooting, see [Setup and Support](setup-and-support.md).

## What It Automates

The tool is meant to remove most of the repetitive masking work from face anonymization in Kdenlive. Instead of keyframing masks by hand, you copy the clip XML, let the tool generate tracked mask effects, then polish the handful of frames where tracking is still imperfect.

`kdenlive_face_mask.py` targets the XML snippet you get when you copy a clip in Kdenlive. It resolves the source media path from embedded bin data, runs offline InsightFace detection over the clip frame range, and injects `mask_start-frei0r.alphaspot` effects for tracked faces.

For Kdenlive compatibility, non-overlapping spans for the same tracked face are collapsed into a single mask effect with zero-size gap keyframes between spans. Only overlapping faces are emitted as separate mask effects.

Position smoothing is motion-adaptive: slow movement is smoothed to reduce jitter, while sharp jumps keep the real detected position so the mask does not visibly trail fast head movement.

Mask tilt follows the detected eye-line angle for each tracked face. `--tilt` remains the neutral baseline, so you can bias the generated mask orientation while still keeping per-frame face rotation.

Detection and tracking run at full clip FPS. Keyframe-density controls only reduce serialized Kdenlive keyframes written into XML after tracking is complete.

## Basic Usage

Clipboard-driven usage:

```bash
uv run kdenlive-face-mask --clipboard-in --clipboard-out
```

File-based input/output:

```bash
uv run kdenlive-face-mask /path/to/copied-clip.xml -o /path/to/rewritten-clip.xml
```

The input path is a placeholder for a real copied-clip XML file you saved locally; it is not a file shipped in this repository.

Pipe-based usage:

```bash
wl-paste --no-newline | uv run kdenlive-face-mask > rewritten-clip.xml
```

## Useful Flags

- `input` (positional) reads copied-clip XML from a file path; omit it to read from stdin.
- `-o, --output FILE` writes rewritten XML to a file; omit it to write to stdout.
- `--clipboard-in` and `--clipboard-out` read/write XML through the system clipboard. Requires `wl-clipboard` or `xclip`/`xsel` on Linux, uses built-in `pbpaste`/`pbcopy` on macOS, and prefers a UTF-8 PowerShell path on Windows with `clip.exe` as fallback.
- `--det-size WIDTHxHEIGHT` sets detector input size, for example `320x320` for speed or `640x640` for smaller faces.
- `--model-name MODEL` chooses InsightFace model, for example `buffalo_s` for speed or `buffalo_l` for offline quality.
- `--provider-mode auto|cuda|rocm|coreml|openvino|migraphx|cpu`.
- `auto` tries CUDA, ROCm, CoreML, and OpenVINO in that order before falling back to CPU.
- Use `coreml` on Apple Silicon for GPU-accelerated inference after installing the `cpu` extra.
- Use `openvino` on Intel hardware with the `openvino` extra installed.
- Use `migraphx` only if you explicitly want MIGraphX.
- `--process-width 0` keeps full source resolution during detection.
- `--keyframe-fps` limits emitted Kdenlive keyframes while preserving full-rate tracking internally.
- `--adaptive-keyframes` varies emitted keyframe density by motion speed.
- `--keep-existing-masks` appends generated masks instead of replacing existing `mask_start` effects.

## Advanced Flags

- `--min-score` drops low-confidence face detections before tracking.
- `--pad-x` and `--pad-y` widen the generated mask around the face box.
- `--smooth-window` controls temporal smoothing radius.
- `--min-keyframe-fps` and `--max-keyframe-fps` bound adaptive output density.
- `--max-gap` controls how many missing detection frames can be bridged inside one track.
- `--min-track-length` discards very short false-positive tracks.
- `--shape` sets the generated alphaspot mask geometry.
- `--tilt` sets the neutral alphaspot tilt baseline before tracked face-angle offsets are applied.
- `--progress-every N` logs progress every N processed frames.
- `--log-level DEBUG|INFO|WARNING|ERROR` controls logging verbosity.

## Example Commands

Quality-first clipboard run:

```bash
uv run kdenlive-face-mask --clipboard-in --clipboard-out --det-size 640x640 --process-width 0 --model-name buffalo_l
```

Faster clipboard run:

```bash
uv run kdenlive-face-mask --clipboard-in --clipboard-out --det-size 320x320 --process-width 256 --model-name buffalo_s
```

Fixed output keyframe rate:

```bash
uv run kdenlive-face-mask --clipboard-in --clipboard-out --keyframe-fps 15
```

Adaptive output keyframe rate:

```bash
uv run kdenlive-face-mask --clipboard-in --clipboard-out --adaptive-keyframes --min-keyframe-fps 12 --max-keyframe-fps 60
```

Detailed fast local CPU pass:

```bash
uv run kdenlive-face-mask \
  --clipboard-in \
  --clipboard-out \
  --det-size 320x320 \
  --process-width 256 \
  --model-name buffalo_s \
  --provider-mode cpu \
  --pad-x 0.15 \
  --pad-y 0.15 \
  --adaptive-keyframes \
  --min-keyframe-fps 0
```

What this command does:

- Reads copied clip XML from clipboard.
- Writes rewritten clip XML back to clipboard.
- Favors speed with `320x320` and reduced process width.
- Uses faster `buffalo_s` model.
- Forces CPU for predictable local runs.
- Keeps mask tighter with reduced padding.
- Uses adaptive keyframe thinning after full-rate tracking.
- Allows very aggressive thinning in static regions while preserving segment anchors.

Detailed fast local NVIDIA pass:

```bash
uv run --extra cuda kdenlive-face-mask \
  --clipboard-in \
  --clipboard-out \
  --det-size 320x320 \
  --process-width 320 \
  --model-name buffalo_s \
  --provider-mode cuda \
  --pad-x 0.15 \
  --pad-y 0.15 \
  --adaptive-keyframes \
  --min-keyframe-fps 0
```

What this command does:

- Runs the installed CLI with the CUDA extra enabled.
- Requests NVIDIA CUDA first and falls back to CPU if CUDA initialization fails.
- Keeps the rest of the fast local workflow similar to the CPU example.

## Using Generated Masks in Kdenlive

Generated `Alpha shapes (Mask)` effects define where the censor appears. The effect stack should be:

1. `Alpha shapes (Mask)`
2. One or more censor effects
3. `Mask Apply`

In practice, generated face mask effects go above the effect to restrict, and `Mask Apply` goes below.

Behavior notes:

- If blur/pixelize/fill sits between `Alpha shapes (Mask)` and `Mask Apply`, censor applies to the face.
- Subtract-style mask operations invert that logic.
- Generator emits first mask with `Operation = Write on clear` and later masks with `Operation = Max` so multiple masks can share one censor chain.

Privacy note: blur is not guaranteed anonymization. Depending on the footage and your threat model, Gaussian blur can be insufficient and may be reversible in some cases. Use stronger obscuration when you need real anonymity rather than a simple visual softening effect.

### Blur Tutorial

1. Generate and paste tracked clip so it contains `Alpha shapes (Mask)`.
2. Open the clip effect stack.
3. Keep generated mask rows together at top of the masked block.
4. Add blur directly below last generated mask row.
5. Increase blur strength as needed.
6. Add `Mask Apply` directly below blur.
7. Scrub timeline and confirm blur is only inside tracked face mask.

Recommended blur effects: `Blur (GPU)`, `Gaussian Blur`, `Average Blur`, `Planes Blur`, or `Directional Blur`.

Effect stack example:

```text
Alpha shapes (Mask)
Alpha shapes (Mask)
Alpha shapes (Mask)
Blur (GPU)
Mask Apply
```

If blur affects full frame, reorder so blur is between `Alpha shapes (Mask)` and `Mask Apply`.

### Pixelation Tutorial

1. Generate and paste tracked clip with `Alpha shapes (Mask)` rows.
2. Add `Pixelize (advanced)` below last generated mask row.
3. Increase block width and height until face is obscured.
4. Add `Mask Apply` below pixelize effect.
5. Scrub timeline to verify pixelation is face-only.

`Obscure` can be used as a simpler, less tunable alternative.

### Solid Black Or Solid Color Tutorial

1. Generate and paste tracked clip with `Alpha shapes (Mask)` rows.
2. Add `Draw Box` below last generated mask row.
3. Set color.
4. Set `Top-left X` and `Top-left Y` to `0`, and size box to full frame.
5. Raise thickness until effectively filled.
6. Optionally enable replace color/alpha.
7. Add `Mask Apply` below `Draw Box`.
8. Scrub timeline to verify fill is face-only.

Effect stack example:

```text
Alpha shapes (Mask)
Draw Box
Mask Apply
```

Full-frame box is intentional: mask defines final visible censor shape.

### Multiple Masks On One Clip

If multiple `Alpha shapes (Mask)` effects are generated, usually keep all generated mask rows together, then place one shared censor effect and one shared `Mask Apply` below them.

Example shared blur chain:

```text
Alpha shapes (Mask)
Alpha shapes (Mask)
Alpha shapes (Mask)
Blur
Mask Apply
```

This works because later mask rows use `Operation = Max`, so mask shapes accumulate.

Use separate censor chains only when intentionally applying different styles or strengths per face.

If masks were generated with an older tool version, manually set every generated `Alpha shapes (Mask)` after the first to `Operation = Max` before using the shared-chain workflow.
