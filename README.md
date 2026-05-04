# Kdenlive Face Tracking Tool

Standalone offline tooling for Kdenlive mask generation.

Targets Linux, macOS, and Windows, with CPU as the default path and optional CUDA, ROCm, CoreML, and OpenVINO providers where the host runtime stack supports them.

## Install

Default CPU install:

```bash
uv sync --extra cpu
```

NVIDIA CUDA install:

```bash
uv sync --extra cuda
```

AMD ROCm install:

```bash
uv sync --extra rocm
```

Intel OpenVINO install:

```bash
uv sync --extra openvino
```

The installed CLI is:

```bash
uv run kdenlive-face-mask --help
```

Why four install modes:

- `uv sync --extra cpu` installs the standard `onnxruntime` package for CPU mode and CoreML-capable macOS hosts.
- `uv sync --extra cuda` adds NVIDIA CUDA ONNX Runtime packages. NVIDIA users usually do not need the container workflow.
- `uv sync --extra rocm` adds ROCm ONNX Runtime packages for users who want AMD GPU inference.
- `uv sync --extra openvino` adds OpenVINO ONNX Runtime packages for Intel GPU inference.
- CoreML (Apple Silicon and Intel Mac GPU) is provided by the standard `onnxruntime` package, so it uses the same `cpu` extra.
- Choose exactly one runtime extra per environment. The ONNX Runtime packages are alternatives, not meant to be installed together.
- The tool behavior is unchanged either way: provider selection is controlled at runtime with `--provider-mode`.

## Recommended First Run

For a first run on a local desktop machine, use the easiest path first:

1. `uv sync --extra cpu`
2. `uv run kdenlive-face-mask --doctor`
3. `uv run kdenlive-face-mask --clipboard-in --clipboard-out --provider-mode cpu`
4. After the CPU path works, try `cuda`, `rocm`, `coreml`, `openvino`, or the ROCm container workflow if you want acceleration.

Replace step 1 with the runtime extra you actually want to test (`cpu`, `cuda`, `rocm`, or `openvino`).

If clipboard integration is unavailable or you want easier debugging, switch to file-based input/output documented later in the README instead.

## Runtime Notes

- First run may download InsightFace model weights into the local model cache if the selected model is not already present.
- `--clipboard-in` and `--clipboard-out` work on Linux (`wl-paste`/`wl-copy`, `xclip`, or `xsel`), macOS (`pbpaste`/`pbcopy`), and Windows (`powershell.exe`/`clip.exe`).
- If you want the fewest moving parts, start with CPU mode first and then switch to CUDA, ROCm, CoreML, or OpenVINO after the basic workflow works.
- If ONNX Runtime cannot initialize a requested GPU provider, the tool falls back to CPU.

## Validation Status

- Real-world validation so far includes one Arch Linux machine with an AMD 8745HS APU.
- The included ROCm container workflow has also been validated on that host and successfully initialized ROCm, MIGraphX, and CPU providers.
- The repository includes CI smoke coverage for Linux, macOS, and Windows install, unit-test, and CLI-help checks.
- Non-Linux hosts and most GPU/provider combinations should still be treated as expected-to-work rather than broadly validated.
- Intel macOS is not currently supported by the documented install path because the `cpu` extra currently resolves to Apple Silicon macOS wheels, not Intel macOS wheels.

## Development Note

This project was developed with substantial AI assistance, including architecture planning, Kdenlive XML analysis, iterative requirement refinement, code generation, and most of the documentation.

The implementation has since been manually reviewed at a high level and tested in the environments documented here, but it has not been independently re-derived or exhaustively line-by-line audited.

## Compatibility Matrix

| Platform | CPU mode | CUDA | ROCm | CoreML | OpenVINO | Clipboard I/O |
| --- | --- | --- | --- | --- | --- | --- |
| Linux (x86_64) | Validated on one Arch Linux + AMD host | Expected with a matching NVIDIA stack | Expected on compatible AMD Linux stacks | Not applicable | Expected on compatible Intel stacks | Implemented; Linux workflow validated |
| Windows (x86_64) | Expected; CI smoke coverage only | Expected with a matching NVIDIA stack | Not supported | Not applicable | Expected on compatible Intel stacks | Implemented; not hardware-validated |
| macOS (Apple Silicon) | Expected | Not supported | Not supported | Expected | Not supported | Implemented; not hardware-validated |
| macOS (Intel) | Not currently supported by the documented install path | Not supported | Not supported | Not currently supported by the documented install path | Not supported | Command paths are implemented, but runtime dependency support is currently unconfirmed |

Notes:

- CPU mode is the least risky default where the required wheels install successfully.
- GPU/provider availability depends on host drivers, runtime libraries, and ONNX Runtime provider builds.
- The included ROCm container has been observed to initialize ROCm, MIGraphX, and CPU providers on the validated AMD Linux host.
- CI smoke coverage checks install, unit tests, and `kdenlive-face-mask --help` on Linux, macOS, and Windows; it does not prove GPU-backed inference or broad hardware compatibility.

## Doctor Mode

Use the built-in doctor mode to print host diagnostics without processing clip XML:

```bash
uv run kdenlive-face-mask --doctor
```

Doctor mode is the recommended diagnostic path because it checks both what ONNX Runtime advertises and what detector initialization can actually use on the current machine.

Doctor mode reports:

- Host OS, architecture, and Python version.
- ONNX Runtime import status and available execution providers.
- Real provider usability checks by attempting detector initialization for each detected provider mode and CPU fallback.
- Detected clipboard backend commands.
- The current README support bucket for this host.
- A suggested provider mode for first-run and preferred local testing.

Notes:

- Doctor mode can take longer than `--help` because it attempts real detector initialization.
- If the selected InsightFace model is not cached yet, doctor mode may trigger a model download before provider usability checks can complete.

On hosts that appear to work but are still untested in the README, or are not listed there yet, doctor mode also prints a suggested issue/PR title and body. The suggestion is advisory only: support claims are still updated manually after review rather than inferred automatically from command output.

## Low-Level Provider List Check

If you need a lower-level check of what ONNX Runtime advertises before full detector initialization, use:

```bash
uv run python -c "import onnxruntime as ort; print('\\n'.join(ort.get_available_providers()))"
```

This check does not prove that detector initialization succeeds for a provider; use `--doctor` for the stronger check.

Expected examples:

- CPU-only environment: `CPUExecutionProvider`
- NVIDIA environment: includes `CUDAExecutionProvider`
- AMD ROCm environment (Linux): includes `ROCMExecutionProvider`
- Apple Silicon or Intel Mac environment: includes `CoreMLExecutionProvider`
- Intel OpenVINO environment: includes `OpenVINOExecutionProvider`

You can force a provider mode during execution with `--provider-mode` (`cuda`, `rocm`, `coreml`, `openvino`, `migraphx`, or `cpu`). If provider init fails, the tool falls back to CPU.

## kdenlive_face_mask.py

`kdenlive_face_mask.py` targets the XML snippet you get when you copy a clip in Kdenlive. It resolves the source media path from embedded bin data, runs offline InsightFace detection over the clip frame range, and injects `mask_start-frei0r.alphaspot` effects for tracked faces.

For Kdenlive compatibility, non-overlapping spans for the same tracked face are collapsed into a single mask effect with zero-size gap keyframes between spans. Only overlapping faces are emitted as separate mask effects.

Position smoothing is motion-adaptive: slow movement is smoothed to reduce jitter, while sharp jumps keep the real detected position so the mask does not visibly trail fast head movement.

Detection and tracking run at full clip FPS. Keyframe-density controls only reduce serialized Kdenlive keyframes written into XML after tracking is complete.

### Basic Usage

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

### Useful Flags

- `input` (positional) reads copied-clip XML from a file path; omit it to read from stdin.
- `-o, --output FILE` writes rewritten XML to a file; omit it to write to stdout.
- `--clipboard-in` and `--clipboard-out` read/write XML through the system clipboard. Requires `wl-clipboard` or `xclip`/`xsel` on Linux, uses built-in `pbpaste`/`pbcopy` on macOS, and `powershell.exe`/`clip.exe` on Windows.
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

### Advanced Flags

- `--min-score` drops low-confidence face detections before tracking.
- `--pad-x` and `--pad-y` widen the generated mask around the face box.
- `--smooth-window` controls temporal smoothing radius.
- `--min-keyframe-fps` and `--max-keyframe-fps` bound adaptive output density.
- `--max-gap` controls how many missing detection frames can be bridged inside one track.
- `--min-track-length` discards very short false-positive tracks.
- `--shape` and `--tilt` set generated alphaspot mask geometry parameters.
- `--progress-every N` logs progress every N processed frames.
- `--log-level DEBUG|INFO|WARNING|ERROR` controls logging verbosity.

### Example Commands

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

## Troubleshooting

- If first run fails while fetching models, retry on a machine with network access first so InsightFace can populate its cache.
- If `--clipboard-in` or `--clipboard-out` fails, switch to file input/output and confirm your clipboard utility is installed.
- If the tool reports detector initialization failure for CUDA, ROCm, CoreML, OpenVINO, or MIGraphX, rerun with `--provider-mode cpu` to verify the rest of the workflow.
- If copied XML references media outside the current machine or mount layout, use file-based XML input and correct the source media path in Kdenlive first.
- If ONNX Runtime import fails entirely, install one runtime extra first: `cpu` for CPU/CoreML, `cuda` for NVIDIA, `rocm` for AMD ROCm, or `openvino` for Intel OpenVINO.
- If your expected provider is missing in the self-check output, verify you installed the matching extra for that environment and confirm the host driver/runtime installation.
- On Windows, if clipboard reads fail, check `Get-Clipboard` directly in PowerShell; if clipboard writes fail, verify `clip.exe` is available on `PATH`.

## ROCm Container Workflow For Kdenlive

If ROCm inference is more stable inside a pinned container, use the included ROCm image for offline Kdenlive masking.

Wrapper command options:

- `build` builds the ROCm image referenced by the wrapper script.
- `kdenlive-mask [kdenlive_face_mask args...]` runs this tool inside the ROCm container.
- `--image IMAGE` overrides the image name for either command so you can reuse an image built elsewhere.
- `HSA_OVERRIDE_GFX_VERSION` can be exported on the host before running the wrapper to override the default passthrough value.

Note on image size:

- The first ROCm image build is large (often around 10 GB or more) because the ROCm base/runtime stack is heavy.
- This is normal and mostly unrelated to your source tree size.

Build the shared ROCm image once:

```bash
cd kdenlive-face-tracking-tool
fish run_kdenlive_rocm_container.fish build
```

Reuse an existing image from another project instead of rebuilding:

```bash
fish run_kdenlive_rocm_container.fish --image obs-insightface-sidecar-rocm kdenlive-mask --help
```

Or retag the existing image once so this script uses it by default:

```bash
podman tag obs-insightface-sidecar-rocm kdenlive-face-tracking-rocm
```

The helper includes `kdenlive-mask`, which runs `kdenlive_face_mask.py` in the ROCm image.

Recommended file-based workflow:

```bash
wl-paste --no-newline --type 'text/plain;charset=utf-8' > /tmp/kdenlive-clip.xml

fish run_kdenlive_rocm_container.fish kdenlive-mask \
  /tmp/kdenlive-clip.xml \
  -o /tmp/kdenlive-clip-masked.xml \
  --det-size 320x320 \
  --process-width 256 \
  --model-name buffalo_s \
  --pad-x 0.15 \
  --pad-y 0.15 \
  --adaptive-keyframes \
  --min-keyframe-fps 0

wl-copy < /tmp/kdenlive-clip-masked.xml
```

Recommended pipe-based workflow:

```bash
set -l masked_xml /tmp/kdenlive-clip-masked.xml

wl-paste --no-newline --type 'text/plain;charset=utf-8' \
  | fish run_kdenlive_rocm_container.fish kdenlive-mask \
    --det-size 320x320 \
    --process-width 320 \
    --model-name buffalo_l \
    --pad-x 0.1 \
    --pad-y 0.05 \
    --adaptive-keyframes \
    --min-keyframe-fps 5 \
    > "$masked_xml"

and test -s "$masked_xml"
and wl-copy < "$masked_xml"
```

Why the guard matters:

- `kdenlive-mask` reads XML from stdin and writes rewritten XML to stdout.
- If masking fails, the redirected output file can still be created or truncated.
- `and test -s ...` prevents copying an empty file to clipboard.
- NVIDIA users normally do not need this container path; prefer the CUDA extra above first.

Important container notes:

- Uses ROCm runtime settings with `HSA_OVERRIDE_GFX_VERSION` passthrough.
- Defaults to `--provider-mode rocm` inside the container; pass `--provider-mode cpu` after `kdenlive-mask` to override.
- Pipe workflow needs stdin open, so `kdenlive-mask` runs Podman with `--interactive`.
- Home directory is bind-mounted at the same path, so absolute media paths in copied XML usually resolve without edits.
- `/tmp` is bind-mounted, matching the examples above.
- Keep clipboard utilities on host; container does not need Wayland clipboard access.
- If source media is outside home or `/tmp`, add a bind mount or move/symlink media into a mounted path.

## Turning Masks Into A Censor In Kdenlive

Generated `Alpha shapes (Mask)` effects define where the censor appears. The effect stack should be:

1. `Alpha shapes (Mask)`
2. One or more censor effects
3. `Mask Apply`

In practice, generated face mask effects go above the effect to restrict, and `Mask Apply` goes below.

Behavior notes:

- If blur/pixelize/fill sits between `Alpha shapes (Mask)` and `Mask Apply`, censor applies to the face.
- Subtract-style mask operations invert that logic.
- Generator emits first mask with `Operation = Write on clear` and later masks with `Operation = Max` so multiple masks can share one censor chain.

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

## Current Limitations

- Only copied clip snippets rooted at `<kdenlive-scene>` are supported.
- Only clips with `speed=1` are supported.
- Tool only generates mask effects; blur/pixelize/fill effects are added manually in Kdenlive.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
