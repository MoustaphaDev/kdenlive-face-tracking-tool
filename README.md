# Kdenlive Face Tracking Tool

Automate face-mask tracking in Kdenlive so you spend less time keyframing by hand.

If Kdenlive's built-in motion tracking does not stay glued to faces well enough for your footage, or manual masking would take forever, this tool gives you a fast first pass. It generates tracked `Alpha shapes (Mask)` effects from copied clip XML so you can polish a few misses instead of animating every face from scratch.

The workflow is intentionally simple: copy a clip in Kdenlive, run the tool on that copied XML, paste the generated effects back, then add blur, pixelize, or another censor effect below the masks. Under the hood it runs InsightFace offline over the clip frames, calculates smoothed mask coordinates, and rewrites the copied XML for you.

Designed for Linux, macOS, and Windows, with CPU as the default path and optional CUDA, ROCm, CoreML, and OpenVINO providers where the host runtime stack supports them.

Real-world validation so far is limited to one Arch Linux machine with an AMD 8745HS APU and the included ROCm container workflow. Other platforms and provider combinations should still be treated as expected-to-work rather than broadly validated.

## Demo

<video src="media/demo.mp4" controls muted playsinline loop></video>

If the embedded player does not render in your viewer, open [media/demo.mp4](media/demo.mp4) directly.

## What It's For

- Blurring or censoring a lot of faces without hand-keyframing every mask.
- Recovering when the native tracker is not sticky enough for face work.
- Generating a usable first pass that you can finish by tweaking only the bad frames.
- Staying inside a local, offline Kdenlive workflow instead of shipping media to an external service.

## Workflow

1. Copy a clip in Kdenlive.
2. Run `kdenlive-face-mask --clipboard-in --clipboard-out`; that clipboard workflow reads the copied XML, tracks the faces, injects smoothed mask keyframes, and writes the rewritten XML back to your clipboard.
3. Paste the rewritten clip or effects back into Kdenlive, place your censor effect between the generated masks and `Mask Apply`, and keep editing.
4. If one face drifts for a few frames, fix just those keyframes instead of building the whole mask animation by hand.

## Install

Start with the simplest CPU path:

```bash
uv sync --extra cpu
uv run kdenlive-face-mask --help
```

For CUDA, ROCm, OpenVINO, platform notes, diagnostics, and the ROCm container workflow, see [Setup and Support](docs/setup-and-support.md).

## Quick Start

Check the local environment first:

```bash
uv run kdenlive-face-mask --doctor
```

Process copied clip XML through the clipboard with the CPU path:

```bash
uv run kdenlive-face-mask --clipboard-in --clipboard-out --provider-mode cpu
```

If clipboard integration is unavailable or you want a simpler debugging path, use file input and output instead:

```bash
uv run kdenlive-face-mask /path/to/copied-clip.xml -o /path/to/rewritten-clip.xml
```

For install variants, provider tuning, and troubleshooting, see [Setup and Support](docs/setup-and-support.md). For flags, examples, and Kdenlive workflow details, see [Usage Guide](docs/usage.md).

## What You Get

- One mask effect per simultaneously visible face.
- Non-overlapping spans for the same face collapsed into a single effect with zero-size gap anchors.
- Motion-adaptive smoothing to reduce jitter without visibly trailing fast movement.
- Face-angle-driven tilt keyframes, using `--tilt` as the neutral baseline.
- Tunable detection and tracking behavior: model choice, detection size, score threshold, padding, gap handling, smoothing, and fixed or adaptive keyframe density.
- Successful no-face runs: by default generated mask effects are removed, and `--keep-existing-masks` preserves any existing ones.
- A practical first pass, not a promise of perfect tracking on every difficult shot.

## Use It In Kdenlive

After pasting the rewritten clip back into Kdenlive, put your censor effect between the generated masks and `Mask Apply`.

Example stack:

```text
Alpha shapes (Mask)
Blur / Pixelize / Draw Box
Mask Apply
```

Privacy note: blur is not guaranteed anonymization. Depending on the footage and your threat model, Gaussian blur can be insufficient and may be reversible in some cases. Use stronger obscuration when you need real anonymity rather than a simple visual softening effect.

For multi-mask behavior, effect-stack tutorials, and example censor workflows, see [Usage Guide](docs/usage.md).

<details>
<summary>AI Development Note</summary>

This project was developed with substantial AI assistance, including architecture planning, Kdenlive XML analysis, iterative requirement refinement, code generation, and most of the documentation.

The implementation has since been manually reviewed at a high level, but it has not been independently re-derived or exhaustively line-by-line audited.

Real-world testing so far is limited to my own machine, an AMD 8745HS APU system, plus the included ROCm container workflow. Broader platform and provider expectations are based on the implemented code paths, dependency support, and (not yet passing) CI smoke coverage rather than direct hands-on validation across every environment.

</details>

## Learn More

- [Setup and Support](docs/setup-and-support.md): install modes, compatibility matrix, doctor mode, troubleshooting, and ROCm container workflow.
- [Usage Guide](docs/usage.md): CLI workflows, flags, example commands, and Kdenlive masking tutorials.
- [Release Checklist](RELEASE.md): maintainer release steps.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
