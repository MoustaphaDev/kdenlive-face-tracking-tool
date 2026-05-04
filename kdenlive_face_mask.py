#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import heapq
import io
import logging
import math
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG = logging.getLogger("kdenlive-face-mask")

MASK_EFFECT_ID = "mask_start-frei0r.alphaspot"
DEFAULT_SHAPE = 0.38
DEFAULT_TILT = 0.5
DEFAULT_PROGRESS_EVERY = 120
DEFAULT_MIN_KEYFRAME_FPS = 12.0
ADAPTIVE_KEYFRAME_MOTION_RATIO_LOW = 0.012
ADAPTIVE_KEYFRAME_MOTION_RATIO_HIGH = 0.05
MASK_OPERATION_WRITE_ON_CLEAR = "0"
MASK_OPERATION_MAX = "0.3"
MODERATE_POSITION_MOTION_RATIO = 0.06
FAST_POSITION_MOTION_RATIO = 0.12

READ_CLIPBOARD_COMMANDS = [
    ["wl-paste", "--no-newline", "--type", "text/plain;charset=utf-8"],
    ["wl-paste", "--no-newline", "--type", "text/plain"],
    ["wl-paste", "--no-newline"],
    ["xclip", "-selection", "clipboard", "-out"],
    ["xsel", "--clipboard", "--output"],
    ["pbpaste"],
    [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); $OutputEncoding = [System.Text.UTF8Encoding]::new($false); Get-Clipboard -Raw",
    ],
]

WRITE_CLIPBOARD_COMMANDS = [
    ["wl-copy"],
    ["xclip", "-selection", "clipboard", "-in"],
    ["xsel", "--clipboard", "--input"],
    ["pbcopy"],
    [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false); Set-Clipboard -Value ([Console]::In.ReadToEnd())",
    ],
    ["clip.exe"],
]

PROVIDER_MODE_TO_ORT_PROVIDER = {
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

DOCTOR_PROVIDER_PROBE_ORDER = ["cuda", "rocm", "coreml", "openvino", "migraphx", "cpu"]


@dataclass
class SceneContext:
    root: ET.Element
    video_clip: ET.Element
    source_path: Path
    fps: float
    in_frame: int
    out_frame: int
    total_frames: int
    frame_width: int | None
    frame_height: int | None


@dataclass(frozen=True)
class FaceBox:
    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float = 0.0
    score: float = 1.0

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)


@dataclass(frozen=True)
class TrackSample:
    frame_index: int
    cx: float
    cy: float
    half_w: float
    half_h: float
    angle_deg: float = 0.0
    synthetic: bool = False

    def to_box(self) -> FaceBox:
        return FaceBox(
            x1=self.cx - self.half_w,
            y1=self.cy - self.half_h,
            x2=self.cx + self.half_w,
            y2=self.cy + self.half_h,
            angle_deg=self.angle_deg,
        )


@dataclass
class FaceTrack:
    track_id: int
    samples: list[TrackSample] = field(default_factory=list)
    vx: float = 0.0
    vy: float = 0.0
    miss_count: int = 0

    def first_frame(self) -> int:
        return self.samples[0].frame_index

    def last_frame(self) -> int:
        return self.samples[-1].frame_index

    def average_cx(self) -> float:
        return sum(sample.cx for sample in self.samples) / max(1, len(self.samples))

    def real_sample_count(self) -> int:
        return sum(1 for sample in self.samples if not sample.synthetic)

    def predict_sample(self, frame_index: int) -> TrackSample:
        last = self.samples[-1]
        frame_delta = max(1, frame_index - last.frame_index)
        return TrackSample(
            frame_index=frame_index,
            cx=last.cx + self.vx * frame_delta,
            cy=last.cy + self.vy * frame_delta,
            half_w=last.half_w,
            half_h=last.half_h,
            angle_deg=last.angle_deg,
            synthetic=True,
        )

    def append_sample(self, sample: TrackSample) -> None:
        if self.samples:
            previous = self.samples[-1]
            frame_delta = max(1, sample.frame_index - previous.frame_index)
            measured_vx = (sample.cx - previous.cx) / frame_delta
            measured_vy = (sample.cy - previous.cy) / frame_delta
            blend = 0.4 if not sample.synthetic else 0.2
            self.vx = (1.0 - blend) * self.vx + blend * measured_vx
            self.vy = (1.0 - blend) * self.vy + blend * measured_vy
        self.samples.append(sample)
        self.miss_count = 0 if not sample.synthetic else self.miss_count + 1

    def trim_trailing_synthetic(self) -> None:
        while self.samples and self.samples[-1].synthetic:
            self.samples.pop()


@dataclass(frozen=True)
class MaskFrame:
    frame_index: int
    pos_x: float
    pos_y: float
    size_x: float
    size_y: float
    tilt: float = DEFAULT_TILT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tracked Kdenlive mask effects from a copied clip XML snippet.",
    )
    parser.add_argument("input", nargs="?", help="Input Kdenlive copied-clip XML file. Defaults to stdin.")
    parser.add_argument("-o", "--output", help="Write the modified XML snippet to this file instead of stdout.")
    parser.add_argument(
        "--clipboard-in",
        action="store_true",
        help="Read the copied clip XML from the system clipboard.",
    )
    parser.add_argument(
        "--clipboard-out",
        action="store_true",
        help="Write the modified XML snippet back to the system clipboard.",
    )
    parser.add_argument("--model-name", default="buffalo_l", help="InsightFace model name to load.")
    parser.add_argument(
        "--provider-mode",
        default="auto",
        choices=("auto", "cuda", "rocm", "migraphx", "coreml", "openvino", "cpu"),
        help="ONNX Runtime provider preference. auto tries CUDA, ROCm, CoreML, OpenVINO, then CPU.",
    )
    parser.add_argument(
        "--det-size",
        default="640x640",
        help="Detection network size as WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=0,
        help="Resize frames to this width before detection. 0 keeps source resolution.",
    )
    parser.add_argument("--min-score", type=float, default=0.45, help="Drop detections below this confidence score.")
    parser.add_argument("--pad-x", type=float, default=0.25, help="Horizontal mask padding as a fraction of box width.")
    parser.add_argument("--pad-y", type=float, default=0.35, help="Vertical mask padding as a fraction of box height.")
    parser.add_argument("--max-gap", type=int, default=4, help="Maximum missing-frame gap to bridge within a track.")
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=4,
        help="Discard tracks with fewer real detections than this.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=2,
        help="Radius of the temporal smoothing window applied to each track.",
    )
    parser.add_argument(
        "--keyframe-fps",
        type=float,
        default=0.0,
        help="Limit emitted Kdenlive keyframes to approximately this FPS. 0 keeps one keyframe per tracked frame.",
    )
    parser.add_argument(
        "--adaptive-keyframes",
        action="store_true",
        help="Vary emitted keyframe density by motion speed while preserving full-rate tracking internally.",
    )
    parser.add_argument(
        "--min-keyframe-fps",
        type=float,
        default=DEFAULT_MIN_KEYFRAME_FPS,
        help="Minimum emitted keyframe FPS when adaptive keyframes are enabled.",
    )
    parser.add_argument(
        "--max-keyframe-fps",
        type=float,
        default=0.0,
        help="Maximum emitted keyframe FPS when adaptive keyframes are enabled. 0 uses the clip FPS.",
    )
    parser.add_argument(
        "--shape",
        type=float,
        default=DEFAULT_SHAPE,
        help="alphaspot shape value to write into generated mask effects.",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=DEFAULT_TILT,
        help="alphaspot tilt value to use for generated mask effects.",
    )
    parser.add_argument(
        "--keep-existing-masks",
        action="store_true",
        help="Append generated masks instead of replacing existing mask_start effects.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Log detection progress every N processed frames.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print environment diagnostics and exit without processing clip XML.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.process_width < 0:
        raise ValueError(f"Invalid process-width {args.process_width}; expected 0 or a positive integer")
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError(f"Invalid min-score {args.min_score}; expected a value between 0 and 1")
    if args.pad_x <= -1.0:
        raise ValueError(f"Invalid pad-x {args.pad_x}; expected a value greater than -1")
    if args.pad_y <= -1.0:
        raise ValueError(f"Invalid pad-y {args.pad_y}; expected a value greater than -1")
    if args.max_gap < 0:
        raise ValueError(f"Invalid max-gap {args.max_gap}; expected 0 or a positive integer")
    if args.min_track_length < 1:
        raise ValueError(f"Invalid min-track-length {args.min_track_length}; expected at least 1")
    if args.smooth_window < 0:
        raise ValueError(f"Invalid smooth-window {args.smooth_window}; expected 0 or a positive integer")
    if args.keyframe_fps < 0.0:
        raise ValueError(f"Invalid keyframe-fps {args.keyframe_fps}; expected 0 or a positive number")
    if args.min_keyframe_fps < 0.0:
        raise ValueError(f"Invalid min-keyframe-fps {args.min_keyframe_fps}; expected 0 or a positive number")
    if args.max_keyframe_fps < 0.0:
        raise ValueError(f"Invalid max-keyframe-fps {args.max_keyframe_fps}; expected 0 or a positive number")
    if args.max_keyframe_fps > 0.0 and args.min_keyframe_fps > args.max_keyframe_fps:
        raise ValueError(
            f"Invalid adaptive keyframe range min={args.min_keyframe_fps} max={args.max_keyframe_fps}; expected min <= max"
        )
    if args.progress_every < 0:
        raise ValueError(f"Invalid progress-every {args.progress_every}; expected 0 or a positive integer")


def configure_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, level_name), format="[maskgen] %(levelname)s: %(message)s")


def installed_package_version() -> str:
    try:
        return metadata.version("kdenlive-face-tracking-tool")
    except metadata.PackageNotFoundError:
        return "0+unknown"


def unique_available_commands(commands: Sequence[Sequence[str]]) -> list[str]:
    available: list[str] = []
    seen: set[str] = set()
    for command in commands:
        command_name = command[0]
        if command_name in seen:
            continue
        if shutil.which(command_name) is None:
            continue
        available.append(command_name)
        seen.add(command_name)
    return available


def detect_clipboard_support() -> dict[str, Any]:
    read_commands = unique_available_commands(READ_CLIPBOARD_COMMANDS)
    write_commands = unique_available_commands(WRITE_CLIPBOARD_COMMANDS)
    return {
        "read_command": read_commands[0] if read_commands else None,
        "write_command": write_commands[0] if write_commands else None,
        "available_read_commands": read_commands,
        "available_write_commands": write_commands,
    }


def detect_host_environment() -> dict[str, str | None]:
    system_name = platform.system() or sys.platform
    machine = platform.machine() or "unknown"
    distribution = None
    distribution_id = None

    if system_name == "Linux":
        try:
            os_release = platform.freedesktop_os_release()
        except Exception:
            os_release = {}
        distribution = os_release.get("PRETTY_NAME") or os_release.get("NAME")
        distribution_id = os_release.get("ID")

    return {
        "system": system_name,
        "machine": machine,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "distribution": distribution,
        "distribution_id": distribution_id,
    }


def detect_onnxruntime_support() -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except (ImportError, OSError) as exc:
        return {
            "import_ok": False,
            "version": None,
            "available_providers": [],
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:
        return {
            "import_ok": True,
            "version": getattr(ort, "__version__", None),
            "available_providers": [],
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    return {
        "import_ok": True,
        "version": getattr(ort, "__version__", None),
        "available_providers": providers,
        "error": None,
    }


def preferred_provider_mode(available_providers: Sequence[str]) -> str | None:
    provider_order = [
        ("CUDAExecutionProvider", "cuda"),
        ("ROCMExecutionProvider", "rocm"),
        ("CoreMLExecutionProvider", "coreml"),
        ("OpenVINOExecutionProvider", "openvino"),
        ("MIGraphXExecutionProvider", "migraphx"),
        ("CPUExecutionProvider", "cpu"),
    ]
    for provider_name, provider_mode in provider_order:
        if provider_name in available_providers:
            return provider_mode
    return None


def doctor_probe_modes(available_providers: Sequence[str]) -> list[str]:
    modes: list[str] = []
    for provider_mode in DOCTOR_PROVIDER_PROBE_ORDER:
        provider_name = PROVIDER_MODE_TO_ORT_PROVIDER[provider_mode]
        if provider_name in available_providers:
            modes.append(provider_mode)
    return modes


def probe_provider_mode(det_size: tuple[int, int], model_name: str, provider_mode: str) -> dict[str, Any]:
    requested_provider = PROVIDER_MODE_TO_ORT_PROVIDER[provider_mode]
    previous_level = LOG.level
    try:
        # Doctor probes are expected to encounter unusable provider setups;
        # keep those failures in the report instead of warning on stderr.
        LOG.setLevel(max(previous_level, logging.ERROR))
        with contextlib.redirect_stderr(io.StringIO()):
            _detector, active_providers, _cv2 = build_detector(det_size, model_name, provider_mode)
    except Exception as exc:
        return {
            "mode": provider_mode,
            "requested_provider": requested_provider,
            "status": "failed",
            "active_providers": [],
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        LOG.setLevel(previous_level)

    if not active_providers:
        status = "initialized-provider-unknown"
    elif requested_provider in active_providers:
        status = "usable"
    elif provider_mode != "cpu" and "CPUExecutionProvider" in active_providers:
        status = "fallback-to-cpu"
    else:
        status = "initialized-with-different-provider"

    return {
        "mode": provider_mode,
        "requested_provider": requested_provider,
        "status": status,
        "active_providers": list(active_providers),
        "error": None,
    }


def probe_provider_usability(det_size: tuple[int, int], model_name: str, available_providers: Sequence[str]) -> list[dict[str, Any]]:
    return [probe_provider_mode(det_size, model_name, provider_mode) for provider_mode in doctor_probe_modes(available_providers)]


def preferred_usable_provider_mode(provider_probes: Sequence[dict[str, Any]]) -> str | None:
    for provider_mode in DOCTOR_PROVIDER_PROBE_ORDER:
        for probe in provider_probes:
            if probe["mode"] == provider_mode and probe["status"] == "usable":
                return provider_mode
    return None


def format_provider_probe_for_report(probe: dict[str, Any]) -> str:
    active_providers = ", ".join(probe["active_providers"]) if probe["active_providers"] else "none"
    if probe["status"] == "usable":
        return f"- {probe['mode']}: usable ({active_providers})"
    if probe["status"] == "fallback-to-cpu":
        return f"- {probe['mode']}: detected but unusable for detector init; fell back to {active_providers}"
    if probe["status"] == "initialized-provider-unknown":
        return f"- {probe['mode']}: initialized, but active providers could not be introspected"
    return f"- {probe['mode']}: failed ({probe['error']})"


def format_provider_probe_for_issue_body(probe: dict[str, Any]) -> str:
    active_providers = ", ".join(probe["active_providers"]) if probe["active_providers"] else "none"
    if probe["status"] == "usable":
        return f"- {probe['mode']}: usable ({active_providers})"
    if probe["status"] == "fallback-to-cpu":
        return f"- {probe['mode']}: detected but unusable for detector init; fell back to {active_providers}"
    if probe["status"] == "initialized-provider-unknown":
        return f"- {probe['mode']}: initialized, but active providers could not be introspected"
    return f"- {probe['mode']}: failed ({probe['error']})"


def support_assessment_for_host(host: dict[str, str | None]) -> dict[str, Any]:
    system_name = (host["system"] or "").lower()
    machine = (host["machine"] or "").lower()

    if system_name == "linux" and machine in {"x86_64", "amd64"}:
        return {
            "readme_label": "Linux (x86_64)",
            "status": "expected-with-partial-real-world-validation",
            "summary": "Documented in the README and partially validated, but broad hardware coverage is still unverified.",
            "should_suggest_report": True,
        }

    if system_name == "windows" and machine in {"x86_64", "amd64"}:
        return {
            "readme_label": "Windows (x86_64)",
            "status": "expected-with-ci-smoke-coverage",
            "summary": "Documented in the README with CI smoke coverage, but not yet hardware-validated by the project.",
            "should_suggest_report": True,
        }

    if system_name == "darwin" and machine in {"arm64", "aarch64"}:
        return {
            "readme_label": "macOS (Apple Silicon)",
            "status": "expected-but-unverified",
            "summary": "Documented in the README as expected to work, but not yet hardware-validated by the project.",
            "should_suggest_report": True,
        }

    if system_name == "darwin" and machine in {"x86_64", "amd64"}:
        return {
            "readme_label": "macOS (Intel)",
            "status": "unsupported-by-documented-install-path",
            "summary": "The README currently treats Intel macOS as unsupported by the documented install path.",
            "should_suggest_report": False,
        }

    return {
        "readme_label": "Not listed in README",
        "status": "not-listed",
        "summary": "This host is not currently described in the README compatibility matrix.",
        "should_suggest_report": True,
    }


def host_display_name(host: dict[str, str | None]) -> str:
    distribution = host.get("distribution")
    if distribution:
        return f"{distribution} ({host['machine']})"
    return f"{host['system']} ({host['machine']})"


def build_report_suggestion(
    host: dict[str, str | None],
    support: dict[str, Any],
    available_providers: Sequence[str],
    provider_probes: Sequence[dict[str, Any]],
) -> dict[str, str] | None:
    if not support.get("should_suggest_report"):
        return None

    recommended_mode = preferred_usable_provider_mode(provider_probes) or preferred_provider_mode(available_providers) or "cpu"
    provider_text = ", ".join(available_providers) if available_providers else "none detected"
    provider_probe_lines = [format_provider_probe_for_issue_body(probe) for probe in provider_probes]
    title = f"validation report: {support['readme_label']} works on {host_display_name(host)}"
    body = "\n".join(
        [
            "## Validation report",
            f"- Host: {host_display_name(host)}",
            f"- Python: {host['python_version']}",
            f"- ONNX Runtime providers: {provider_text}",
            f"- Recommended provider mode on this host: {recommended_mode}",
            "- Validation performed: end-to-end real clip run",
            "",
            "## Provider usability checks",
            *provider_probe_lines,
            "",
            "## Requested follow-up",
            "- [ ] Please confirm whether this should be reflected in the README validation status.",
            "- [ ] Please mention whether this belongs in an issue or a PR.",
            "",
            "## Notes",
            "- Please mention whether CPU mode worked.",
            "- If you validated GPU acceleration, include the provider mode and driver/runtime details.",
            "- Include whether you tested file I/O, clipboard I/O, or both.",
        ]
    )
    return {
        "title": title,
        "body": body,
    }


def build_doctor_report(args: argparse.Namespace) -> dict[str, Any]:
    host = detect_host_environment()
    onnxruntime_support = detect_onnxruntime_support()
    support = support_assessment_for_host(host)
    available_providers: list[str] = list(onnxruntime_support["available_providers"])
    det_size = parse_det_size(args.det_size)
    provider_probes = probe_provider_usability(det_size, args.model_name, available_providers) if onnxruntime_support["import_ok"] else []
    recommended_mode = preferred_usable_provider_mode(provider_probes) or preferred_provider_mode(available_providers) or "cpu"

    return {
        "tool": {
            "name": "kdenlive-face-mask",
            "version": installed_package_version(),
        },
        "host": host,
        "support": support,
        "onnxruntime": onnxruntime_support,
        "provider_probes": provider_probes,
        "clipboard": detect_clipboard_support(),
        "recommendations": {
            "first_run_provider_mode": "cpu",
            "preferred_provider_mode": recommended_mode,
        },
        "contribution": {
            "should_report_if_successful": bool(support.get("should_suggest_report")),
            "suggested_issue_or_pr": build_report_suggestion(host, support, available_providers, provider_probes),
        },
    }


def format_doctor_report(report: dict[str, Any]) -> str:
    tool: dict[str, Any] = report["tool"]
    host: dict[str, str | None] = report["host"]
    support: dict[str, Any] = report["support"]
    onnxruntime_support: dict[str, Any] = report["onnxruntime"]
    provider_probes: list[dict[str, Any]] = list(report["provider_probes"])
    clipboard: dict[str, Any] = report["clipboard"]
    recommendations: dict[str, Any] = report["recommendations"]
    contribution: dict[str, Any] = report["contribution"]
    suggested_report: dict[str, str] | None = contribution["suggested_issue_or_pr"]
    providers: list[str] = list(onnxruntime_support["available_providers"])

    lines = [
        f"{tool['name']} doctor",
        f"Version: {tool['version']}",
        f"Host: {host_display_name(host)}",
        f"Python: {host['python_version']}",
        f"Platform: {host['platform']}",
        f"README bucket: {support['readme_label']}",
        f"Support status: {support['status']}",
        f"Support summary: {support['summary']}",
        f"ONNX Runtime import: {'ok' if onnxruntime_support['import_ok'] else 'failed'}",
        f"ONNX Runtime version: {onnxruntime_support['version'] or 'unknown'}",
        f"Available providers: {', '.join(providers) if providers else 'none'}",
        f"Clipboard read command: {clipboard['read_command'] or 'none'}",
        f"Clipboard write command: {clipboard['write_command'] or 'none'}",
        f"Suggested first run provider mode: {recommendations['first_run_provider_mode']}",
        f"Preferred provider mode on this host: {recommendations['preferred_provider_mode']}",
    ]

    if provider_probes:
        lines.extend(
            [
                "Provider usability checks:",
                *[format_provider_probe_for_report(probe) for probe in provider_probes],
            ]
        )

    if onnxruntime_support["error"]:
        lines.append(f"ONNX Runtime error: {onnxruntime_support['error']}")

    if contribution["should_report_if_successful"] and suggested_report is not None:
        lines.extend(
            [
                "",
                "If you successfully process a real clip on this host and the README still marks it as untested or does not list it, consider opening an issue or a PR.",
                f"Suggested issue/PR title: {suggested_report['title']}",
                "Suggested issue/PR body:",
                suggested_report["body"],
            ]
        )

    return "\n".join(lines)


def parse_det_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid det-size {value!r}; expected WIDTHxHEIGHT")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid det-size {value!r}; dimensions must be positive")
    return width, height


def format_float(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    text = f"{value:.7f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def property_text(element: ET.Element, name: str) -> str | None:
    prop = element.find(f"./property[@name='{name}']")
    if prop is None or prop.text is None:
        return None
    text = prop.text.strip()
    return text or None


def parse_int(value: str | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing {field_name}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Expected integer {field_name}, got {value!r}") from exc


def parse_float(value: str | None, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Missing {field_name}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected float {field_name}, got {value!r}") from exc


def resolve_source_path(root: ET.Element, resource: str, xml_path: Path | None = None) -> Path:
    path = Path(resource).expanduser()
    if path.is_absolute():
        return path
    scene_root = root.attrib.get("root")
    if scene_root:
        return (Path(scene_root).expanduser() / path).resolve()
    if xml_path is not None:
        return (xml_path.expanduser().resolve().parent / path).resolve()
    return path.resolve()


def find_target_clip(root: ET.Element) -> ET.Element:
    main_clip_ids = [clip_id for clip_id in root.attrib.get("mainClip", "").split(";") if clip_id]
    for clip_id in main_clip_ids:
        candidate = root.find(f"./clip[@id='{clip_id}']")
        if candidate is not None and candidate.attrib.get("audioTrack") is None:
            return candidate

    candidates = [
        clip
        for clip in root.findall("./clip")
        if clip.attrib.get("audioTrack") is None and clip.attrib.get("state") != "2"
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one video clip in copied snippet, found {len(candidates)}")
    return candidates[0]


def find_bin_producer(root: ET.Element, bin_id: str) -> ET.Element:
    bin_element = root.find("./bin")
    if bin_element is None:
        raise ValueError("Copied clip XML is missing the <bin> section")

    for producer in list(bin_element):
        if property_text(producer, "kdenlive:id") == bin_id:
            return producer

    raise ValueError(f"No bin producer found for kdenlive:id={bin_id}")


def resolve_scene_context(xml_text: str, xml_path: Path | None = None) -> SceneContext:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML input: {exc}") from exc

    if root.tag != "kdenlive-scene":
        raise ValueError(f"Unsupported root element {root.tag!r}; expected 'kdenlive-scene'")

    clip = find_target_clip(root)
    speed = parse_float(clip.attrib.get("speed", "1.0"), "clip speed")
    if not math.isclose(speed, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Unsupported clip speed {speed}; only speed=1 clips are supported right now")

    bin_id = clip.attrib.get("binid")
    if not bin_id:
        raise ValueError("Target clip is missing binid")

    producer = find_bin_producer(root, bin_id)
    resource = property_text(producer, "resource")
    if resource is None:
        raise ValueError(f"Bin producer for clip {bin_id} is missing its resource property")

    fps = parse_float(root.attrib.get("fps"), "scene fps")
    in_frame = parse_int(clip.attrib.get("in"), "clip in")
    out_frame = parse_int(clip.attrib.get("out"), "clip out")
    if out_frame < in_frame:
        raise ValueError(f"Clip frame range is invalid: in={in_frame}, out={out_frame}")

    frame_width = property_text(producer, "meta.media.width") or property_text(producer, "meta.media.0.codec.width")
    frame_height = property_text(producer, "meta.media.height") or property_text(producer, "meta.media.0.codec.height")

    return SceneContext(
        root=root,
        video_clip=clip,
        source_path=resolve_source_path(root, resource, xml_path),
        fps=fps,
        in_frame=in_frame,
        out_frame=out_frame,
        total_frames=out_frame - in_frame + 1,
        frame_width=int(frame_width) if frame_width is not None else None,
        frame_height=int(frame_height) if frame_height is not None else None,
    )


def box_iou(left: FaceBox, right: FaceBox) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    union_area = left.width * left.height + right.width * right.height - inter_area
    return inter_area / max(union_area, 1.0)


def candidate_cost(track: FaceTrack, detection: FaceBox) -> float | None:
    predicted = track.predict_sample(track.last_frame() + 1).to_box()
    iou = box_iou(predicted, detection)
    center_distance = math.hypot(predicted.cx - detection.cx, predicted.cy - detection.cy)
    scale = max(predicted.diagonal, detection.diagonal, 1.0)
    size_delta = abs(predicted.width - detection.width) / max(predicted.width, detection.width, 1.0)
    if iou < 0.02 and center_distance > scale * 0.75:
        return None
    return (1.0 - iou) + 0.35 * (center_distance / scale) + 0.2 * size_delta


def mask_box_to_sample(frame_index: int, box: FaceBox, pad_x: float, pad_y: float, synthetic: bool = False) -> TrackSample:
    return TrackSample(
        frame_index=frame_index,
        cx=box.cx,
        cy=box.cy,
        half_w=box.width * 0.5 * (1.0 + pad_x),
        half_h=box.height * 0.5 * (1.0 + pad_y),
        angle_deg=box.angle_deg,
        synthetic=synthetic,
    )


def track_detections(
    detections_by_frame: Iterable[tuple[int, list[FaceBox]]],
    *,
    pad_x: float,
    pad_y: float,
    max_gap: int,
    min_track_length: int,
    smooth_window: int,
) -> list[FaceTrack]:
    active_tracks: list[FaceTrack] = []
    finished_tracks: list[FaceTrack] = []
    next_track_id = 1

    def finalize_track(track: FaceTrack) -> None:
        track.trim_trailing_synthetic()
        if track.real_sample_count() < min_track_length or not track.samples:
            return
        finished_tracks.append(smooth_track(track, smooth_window))

    for frame_index, detections in detections_by_frame:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(active_tracks):
            for detection_index, detection in enumerate(detections):
                cost = candidate_cost(track, detection)
                if cost is not None:
                    candidates.append((cost, track_index, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for cost, track_index, detection_index in sorted(candidates):
            if cost > 1.75 or track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = active_tracks[track_index]
            track.append_sample(mask_box_to_sample(frame_index, detections[detection_index], pad_x, pad_y))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        remaining_tracks: list[FaceTrack] = []
        for track_index, track in enumerate(active_tracks):
            if track_index in matched_tracks:
                remaining_tracks.append(track)
                continue
            if track.miss_count < max_gap:
                track.append_sample(track.predict_sample(frame_index))
                remaining_tracks.append(track)
                continue
            finalize_track(track)
        active_tracks = remaining_tracks

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = FaceTrack(track_id=next_track_id)
            next_track_id += 1
            track.append_sample(mask_box_to_sample(frame_index, detection, pad_x, pad_y))
            active_tracks.append(track)

    for track in active_tracks:
        finalize_track(track)

    finished_tracks.sort(key=lambda track: (track.first_frame(), track.average_cx(), track.track_id))
    return finished_tracks


def smooth_track(track: FaceTrack, radius: int) -> FaceTrack:
    if radius <= 0 or len(track.samples) < 3:
        return track

    def sample_scale(sample: TrackSample) -> float:
        return math.hypot(sample.half_w * 2.0, sample.half_h * 2.0)

    def adjacent_motion_ratio(left: TrackSample, right: TrackSample) -> float:
        return math.hypot(left.cx - right.cx, left.cy - right.cy) / max(sample_scale(left), sample_scale(right), 1.0)

    def local_motion_ratio(index: int) -> float:
        start = max(1, index - radius)
        end = min(len(track.samples) - 1, index + radius)
        if start > end:
            return 0.0
        return max(adjacent_motion_ratio(track.samples[step - 1], track.samples[step]) for step in range(start, end + 1))

    def position_radius(index: int) -> int:
        motion_ratio = local_motion_ratio(index)
        if motion_ratio >= FAST_POSITION_MOTION_RATIO:
            return 0
        if motion_ratio >= MODERATE_POSITION_MOTION_RATIO:
            return min(radius, 1)
        return radius

    def window(index: int, active_radius: int) -> tuple[int, int]:
        return max(0, index - active_radius), min(len(track.samples), index + active_radius + 1)

    def weights_for(center_index: int, start: int, end: int, active_radius: int) -> list[int]:
        return [active_radius + 1 - abs(other_index - center_index) for other_index in range(start, end)]

    smoothed_samples: list[TrackSample] = []
    for index, sample in enumerate(track.samples):
        pos_radius = position_radius(index)
        pos_start, pos_end = window(index, pos_radius)
        pos_weighted = track.samples[pos_start:pos_end]
        pos_weights = weights_for(index, pos_start, pos_end, pos_radius)
        pos_total_weight = float(sum(pos_weights))

        size_start, size_end = window(index, radius)
        size_weighted = track.samples[size_start:size_end]
        size_weights = weights_for(index, size_start, size_end, radius)
        size_total_weight = float(sum(size_weights))

        smoothed_samples.append(
            TrackSample(
                frame_index=sample.frame_index,
                cx=sum(item.cx * weight for item, weight in zip(pos_weighted, pos_weights)) / pos_total_weight,
                cy=sum(item.cy * weight for item, weight in zip(pos_weighted, pos_weights)) / pos_total_weight,
                half_w=sum(item.half_w * weight for item, weight in zip(size_weighted, size_weights)) / size_total_weight,
                half_h=sum(item.half_h * weight for item, weight in zip(size_weighted, size_weights)) / size_total_weight,
                angle_deg=sample.angle_deg,
                synthetic=sample.synthetic,
            )
        )

    track.samples = smoothed_samples
    return track


def _import_detection_modules():
    try:
        import cv2  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("opencv-python is required to read the source media") from exc

    try:
        import numpy as np  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("numpy is required to warm up InsightFace") from exc

    try:
        import onnxruntime as ort  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("onnxruntime is required to run InsightFace") from exc

    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("insightface is required to run face detection") from exc

    return cv2, np, ort, FaceAnalysis


def unique_provider_names(providers: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for provider in providers:
        provider_name = str(provider)
        if provider_name in seen:
            continue
        unique.append(provider_name)
        seen.add(provider_name)
    return unique


def detector_active_providers(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> list[str]:
    if value is None or _depth > 4:
        return []

    if _seen is None:
        _seen = set()

    value_id = id(value)
    if value_id in _seen:
        return []
    _seen.add(value_id)

    get_providers = getattr(value, "get_providers", None)
    if callable(get_providers):
        try:
            providers = get_providers()
        except Exception:
            providers = None
        if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes, bytearray)):
            return unique_provider_names(str(provider) for provider in providers)

    nested_values: list[Any] = []
    for attribute_name in ("models", "model", "det_model", "session", "sess"):
        with contextlib.suppress(Exception):
            nested_value = getattr(value, attribute_name)
            if nested_value is not None:
                nested_values.append(nested_value)

    if isinstance(value, dict):
        nested_values.extend(value.values())
    elif isinstance(value, (list, tuple, set)):
        nested_values.extend(value)
    else:
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            nested_values.extend(vars(value).values())

    providers_found: list[str] = []
    for nested_value in nested_values:
        providers_found.extend(detector_active_providers(nested_value, _seen=_seen, _depth=_depth + 1))
    return unique_provider_names(providers_found)


def build_detector(det_size: tuple[int, int], model_name: str, provider_mode: str):
    cv2, np, ort, FaceAnalysis = _import_detection_modules()
    available = ort.get_available_providers()
    provider_sets: list[list[str]] = []

    if provider_mode == "auto":
        if "CUDAExecutionProvider" in available:
            provider_sets.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
        if "ROCMExecutionProvider" in available:
            provider_sets.append(["ROCMExecutionProvider", "CPUExecutionProvider"])
        if "CoreMLExecutionProvider" in available:
            provider_sets.append(["CoreMLExecutionProvider", "CPUExecutionProvider"])
        if "OpenVINOExecutionProvider" in available:
            provider_sets.append(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "cuda":
        if "CUDAExecutionProvider" in available:
            provider_sets.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "rocm":
        if "ROCMExecutionProvider" in available:
            provider_sets.append(["ROCMExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "migraphx":
        if "MIGraphXExecutionProvider" in available:
            provider_sets.append(["MIGraphXExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "coreml":
        if "CoreMLExecutionProvider" in available:
            provider_sets.append(["CoreMLExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "openvino":
        if "OpenVINOExecutionProvider" in available:
            provider_sets.append(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "cpu":
        provider_sets.append(["CPUExecutionProvider"])
    else:
        raise ValueError(f"Unknown provider-mode {provider_mode!r}")

    last_error: Exception | None = None
    for providers in provider_sets:
        try:
            # InsightFace and ONNX Runtime can print provider/model messages to stdout
            # during initialization. Redirect them to stderr so rewritten XML stdout stays clean.
            with contextlib.redirect_stdout(sys.stderr):
                app = FaceAnalysis(name=model_name, allowed_modules=["detection"], providers=providers)
                app.prepare(ctx_id=0, det_size=det_size)
                app.get(np.zeros((360, 640, 3), dtype=np.uint8))
            return app, detector_active_providers(app), cv2
        except Exception as exc:  # pragma: no cover - hardware/provider dependent
            last_error = exc
            LOG.warning("Detector init failed for providers=%s: %s", providers, exc)

    raise RuntimeError(f"Unable to initialize InsightFace detector: {last_error}")


def angle_from_face(face, scale: float) -> float:
    try:
        keypoints = face.kps
        if keypoints is None or len(keypoints) < 2:
            return 0.0
        lx, ly = float(keypoints[0][0]) * scale, float(keypoints[0][1]) * scale
        rx, ry = float(keypoints[1][0]) * scale, float(keypoints[1][1]) * scale
        return math.degrees(math.atan2(ry - ly, rx - lx))
    except Exception:
        return 0.0


def detect_faces(detector, cv2, frame_bgr, process_width: int, min_score: float) -> list[FaceBox]:
    height, width = frame_bgr.shape[:2]
    scale = 1.0
    detect_frame = frame_bgr
    if process_width > 0 and width > process_width:
        scale = process_width / float(width)
        resized_height = max(1, int(height * scale))
        detect_frame = cv2.resize(frame_bgr, (process_width, resized_height), interpolation=cv2.INTER_AREA)

    inv_scale = 1.0 / scale
    faces = detector.get(detect_frame)
    detections: list[FaceBox] = []
    for face in faces:
        score = float(getattr(face, "det_score", 1.0))
        if score < min_score:
            continue
        x1, y1, x2, y2 = [float(value) * inv_scale for value in face.bbox.tolist()]
        detections.append(
            FaceBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                angle_deg=angle_from_face(face, inv_scale),
                score=score,
            )
        )
    detections.sort(key=lambda item: (item.x1, item.y1))
    return detections


def map_clip_frame_to_source_frame(
    clip_frame: int,
    clip_fps: float,
    source_fps: float,
    max_source_frame_index: int | None = None,
) -> int:
    if clip_fps <= 0.0:
        raise ValueError(f"Invalid clip fps {clip_fps}")
    if source_fps <= 0.0:
        raise ValueError(f"Invalid source fps {source_fps}")

    source_frame = int(math.floor((clip_frame / clip_fps) * source_fps + 1e-9))
    if source_frame < 0:
        return 0
    if max_source_frame_index is not None:
        return min(source_frame, max_source_frame_index)
    return source_frame


def iter_clip_detections(
    context: SceneContext,
    *,
    detector,
    cv2,
    process_width: int,
    min_score: float,
    progress_every: int,
) -> Iterable[tuple[int, list[FaceBox]]]:
    capture = cv2.VideoCapture(str(context.source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source media {context.source_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0.0:
        source_fps = context.fps

    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_source_frame_index = source_frame_count - 1 if source_frame_count > 0 else None

    if context.frame_width is None or context.frame_height is None:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        context.frame_width = width if width > 0 else context.frame_width
        context.frame_height = height if height > 0 else context.frame_height

    cached_source_frame: int | None = None
    cached_frame = None

    try:
        for local_frame in range(context.total_frames):
            clip_frame = context.in_frame + local_frame
            desired_source_frame = map_clip_frame_to_source_frame(
                clip_frame,
                context.fps,
                source_fps,
                max_source_frame_index,
            )

            frame = None
            if cached_frame is not None and cached_source_frame == desired_source_frame:
                frame = cached_frame
            elif cached_source_frame is not None and desired_source_frame == cached_source_frame + 1:
                ok, frame = capture.read()
                if ok:
                    cached_source_frame = desired_source_frame
                    cached_frame = frame
                else:
                    frame = None

            if frame is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, desired_source_frame)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Video read failed at clip frame {clip_frame} (source frame {desired_source_frame})"
                    )
                cached_source_frame = desired_source_frame
                cached_frame = frame

            detections = detect_faces(detector, cv2, frame, process_width, min_score)
            if progress_every > 0 and local_frame > 0 and local_frame % progress_every == 0:
                LOG.info("Processed %s/%s frames", local_frame, context.total_frames)
            yield local_frame, detections
    finally:
        capture.release()


def build_mask_frames(
    track: FaceTrack,
    total_frames: int,
    frame_width: int,
    frame_height: int,
    base_tilt: float = DEFAULT_TILT,
) -> list[MaskFrame]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Invalid source video resolution {frame_width}x{frame_height}; dimensions must be positive")

    sample_by_frame = {sample.frame_index: sample for sample in track.samples}
    sorted_frame_indices = sorted(sample_by_frame)

    def frame_tilt(sample: TrackSample) -> float:
        return (base_tilt + sample.angle_deg / 360.0) % 1.0

    def normalized(sample: TrackSample, frame_index: int | None = None, zero_size: bool = False) -> MaskFrame:
        pos_x = min(max(sample.cx / frame_width, 0.0), 1.0)
        pos_y = min(max(sample.cy / frame_height, 0.0), 1.0)
        size_x = 0.0 if zero_size else min(max(sample.half_w / frame_width, 0.0), 1.0)
        size_y = 0.0 if zero_size else min(max(sample.half_h / frame_height, 0.0), 1.0)
        return MaskFrame(sample.frame_index if frame_index is None else frame_index, pos_x, pos_y, size_x, size_y, frame_tilt(sample))

    segments: list[list[int]] = [[sorted_frame_indices[0]]]
    for frame_index in sorted_frame_indices[1:]:
        if frame_index == segments[-1][-1] + 1:
            segments[-1].append(frame_index)
        else:
            segments.append([frame_index])

    frames: dict[int, MaskFrame] = {}

    for segment in segments:
        segment_start = segment[0]
        segment_end = segment[-1]
        first = sample_by_frame[segment_start]
        last = sample_by_frame[segment_end]

        if segment_start > 0:
            frames[segment_start - 1] = normalized(first, frame_index=segment_start - 1, zero_size=True)

        for frame_index in segment:
            frames[frame_index] = normalized(sample_by_frame[frame_index])

        if segment_end < total_frames - 1:
            frames[segment_end + 1] = normalized(last, frame_index=segment_end + 1, zero_size=True)

    return [frames[index] for index in sorted(frames)]


def merge_disjoint_tracks(tracks: Sequence[FaceTrack]) -> list[FaceTrack]:
    if not tracks:
        return []

    # Reuse mask effects as an interval-partitioning problem so the number of
    # generated effects equals the peak number of simultaneously visible faces.
    merged_tracks: list[FaceTrack] = []
    active_slots: list[tuple[int, int]] = []
    sorted_tracks = sorted(tracks, key=lambda track: (track.first_frame(), track.last_frame(), track.average_cx(), track.track_id))
    for track in sorted_tracks:
        assigned_slot: int | None = None
        if active_slots and active_slots[0][0] < track.first_frame():
            _, assigned_slot = heapq.heappop(active_slots)

        if assigned_slot is None:
            assigned_slot = len(merged_tracks)
            merged_tracks.append(FaceTrack(track_id=assigned_slot + 1, samples=list(track.samples)))
        else:
            merged_tracks[assigned_slot].samples.extend(track.samples)

        heapq.heappush(active_slots, (track.last_frame(), assigned_slot))

    return merged_tracks


def is_zero_size_mask_frame(frame: MaskFrame) -> bool:
    return frame.size_x == 0.0 and frame.size_y == 0.0


def split_visible_mask_segments(frames: Sequence[MaskFrame]) -> list[list[MaskFrame]]:
    segments: list[list[MaskFrame]] = []
    current_segment: list[MaskFrame] = []
    for frame in frames:
        if is_zero_size_mask_frame(frame):
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue
        current_segment.append(frame)
    if current_segment:
        segments.append(current_segment)
    return segments


def mask_frame_motion_ratio(left: MaskFrame, right: MaskFrame) -> float:
    delta_position = math.hypot(right.pos_x - left.pos_x, right.pos_y - left.pos_y)
    delta_size = math.hypot(right.size_x - left.size_x, right.size_y - left.size_y)
    delta_tilt = min(abs(right.tilt - left.tilt), 1.0 - abs(right.tilt - left.tilt)) * 0.25
    scale = max(math.hypot(left.size_x * 2.0, left.size_y * 2.0), math.hypot(right.size_x * 2.0, right.size_y * 2.0), 1e-6)
    return (delta_position + 0.5 * delta_size + delta_tilt) / scale


def compute_segment_motion_ratios(segment: Sequence[MaskFrame]) -> list[float]:
    if len(segment) <= 1:
        return [1.0] * len(segment)

    motion_ratios = [0.0] * len(segment)
    for index in range(1, len(segment)):
        ratio = mask_frame_motion_ratio(segment[index - 1], segment[index])
        motion_ratios[index - 1] = max(motion_ratios[index - 1], ratio)
        motion_ratios[index] = max(motion_ratios[index], ratio)
    return motion_ratios


def resolve_target_keyframe_fps(target_fps: float, clip_fps: float) -> float:
    if target_fps <= 0.0:
        return clip_fps
    return min(target_fps, clip_fps)


def keyframe_interval_frames(clip_fps: float, target_fps: float) -> float:
    resolved_fps = resolve_target_keyframe_fps(target_fps, clip_fps)
    if resolved_fps <= 0.0:
        return 1.0
    return max(1.0, clip_fps / resolved_fps)


def adaptive_target_keyframe_fps(motion_ratio: float, clip_fps: float, min_keyframe_fps: float, max_keyframe_fps: float) -> float:
    resolved_max_fps = resolve_target_keyframe_fps(max_keyframe_fps, clip_fps)
    resolved_min_fps = max(1.0, min(min_keyframe_fps, resolved_max_fps))
    if resolved_min_fps >= resolved_max_fps:
        return resolved_max_fps

    normalized_motion = (motion_ratio - ADAPTIVE_KEYFRAME_MOTION_RATIO_LOW) / max(
        ADAPTIVE_KEYFRAME_MOTION_RATIO_HIGH - ADAPTIVE_KEYFRAME_MOTION_RATIO_LOW,
        1e-6,
    )
    normalized_motion = min(max(normalized_motion, 0.0), 1.0)
    return resolved_min_fps + normalized_motion * (resolved_max_fps - resolved_min_fps)


def select_segment_keyframes(
    segment: Sequence[MaskFrame],
    *,
    clip_fps: float,
    keyframe_fps: float,
    adaptive_keyframes: bool,
    min_keyframe_fps: float,
    max_keyframe_fps: float,
) -> list[MaskFrame]:
    if len(segment) <= 2 or clip_fps <= 0.0:
        return list(segment)

    if not adaptive_keyframes and (keyframe_fps <= 0.0 or keyframe_fps >= clip_fps):
        return list(segment)

    motion_ratios = compute_segment_motion_ratios(segment)
    selected_frames = [segment[0]]
    last_kept_frame_index = segment[0].frame_index

    for index, frame in enumerate(segment[1:-1], start=1):
        if adaptive_keyframes:
            interval = keyframe_interval_frames(
                clip_fps,
                adaptive_target_keyframe_fps(motion_ratios[index], clip_fps, min_keyframe_fps, max_keyframe_fps),
            )
        else:
            interval = keyframe_interval_frames(clip_fps, keyframe_fps)

        if frame.frame_index >= last_kept_frame_index + interval - 1e-9:
            selected_frames.append(frame)
            last_kept_frame_index = frame.frame_index

    if selected_frames[-1].frame_index != segment[-1].frame_index:
        selected_frames.append(segment[-1])
    return selected_frames


def thin_mask_frames(
    frames: Sequence[MaskFrame],
    *,
    clip_fps: float,
    keyframe_fps: float,
    adaptive_keyframes: bool,
    min_keyframe_fps: float,
    max_keyframe_fps: float,
) -> list[MaskFrame]:
    if not frames:
        return []

    effective_keyframe_fps = resolve_target_keyframe_fps(keyframe_fps, clip_fps)
    effective_max_keyframe_fps = resolve_target_keyframe_fps(max_keyframe_fps, clip_fps)
    if not adaptive_keyframes and (effective_keyframe_fps >= clip_fps or keyframe_fps <= 0.0):
        return list(frames)
    if adaptive_keyframes and max(1.0, min_keyframe_fps) >= effective_max_keyframe_fps >= clip_fps:
        return list(frames)

    selected_visible_frame_indices: set[int] = set()
    for segment in split_visible_mask_segments(frames):
        for frame in select_segment_keyframes(
            segment,
            clip_fps=clip_fps,
            keyframe_fps=effective_keyframe_fps,
            adaptive_keyframes=adaptive_keyframes,
            min_keyframe_fps=min_keyframe_fps,
            max_keyframe_fps=effective_max_keyframe_fps,
        ):
            selected_visible_frame_indices.add(frame.frame_index)

    return [frame for frame in frames if is_zero_size_mask_frame(frame) or frame.frame_index in selected_visible_frame_indices]


def build_keyframe_string(frames: Sequence[MaskFrame], selector) -> str:
    return ";".join(f"{frame.frame_index}={format_float(selector(frame))}" for frame in frames)


def build_constant_keyframe_string(frames: Sequence[MaskFrame], value: float) -> str:
    return ";".join(f"{frame.frame_index}={format_float(value)}" for frame in frames)


def make_property(name: str, value: str) -> ET.Element:
    prop = ET.Element("property", {"name": name})
    prop.text = value
    return prop


def make_mask_effect(
    track: FaceTrack,
    total_frames: int,
    frame_width: int,
    frame_height: int,
    clip_fps: float,
    shape: float,
    tilt: float,
    operation: str,
    keyframe_fps: float,
    adaptive_keyframes: bool,
    min_keyframe_fps: float,
    max_keyframe_fps: float,
) -> ET.Element:
    effect = ET.Element("effect", {"id": MASK_EFFECT_ID})
    frames = thin_mask_frames(
        build_mask_frames(track, total_frames, frame_width, frame_height, tilt),
        clip_fps=clip_fps,
        keyframe_fps=keyframe_fps,
        adaptive_keyframes=adaptive_keyframes,
        min_keyframe_fps=min_keyframe_fps,
        max_keyframe_fps=max_keyframe_fps,
    )
    effect.append(make_property("kdenlive:collapsed", "1"))
    effect.append(make_property("filter.Min", build_constant_keyframe_string(frames, 0.0)))
    effect.append(make_property("filter.Transition width", build_constant_keyframe_string(frames, 0.0)))
    effect.append(make_property("filter.Tilt", build_keyframe_string(frames, lambda item: item.tilt)))
    effect.append(make_property("filter.Size Y", build_keyframe_string(frames, lambda item: item.size_y)))
    effect.append(make_property("filter.Max", build_constant_keyframe_string(frames, 1.0)))
    effect.append(make_property("filter.Size X", build_keyframe_string(frames, lambda item: item.size_x)))
    effect.append(make_property("filter.Position Y", build_keyframe_string(frames, lambda item: item.pos_y)))
    effect.append(make_property("filter.Position X", build_keyframe_string(frames, lambda item: item.pos_x)))
    effect.append(make_property("filter.Shape", format_float(shape)))
    effect.append(make_property("filter.Operation", operation))
    return effect


def get_or_create_effects_element(clip: ET.Element) -> ET.Element:
    effects = clip.find("./effects")
    if effects is not None:
        return effects
    return ET.SubElement(clip, "effects", {"parentIn": clip.attrib.get("in", "0")})


def rewrite_scene_with_tracks(
    xml_text: str,
    tracks: Sequence[FaceTrack],
    *,
    frame_width: int,
    frame_height: int,
    shape: float,
    tilt: float,
    replace_existing_masks: bool = True,
    keyframe_fps: float = 0.0,
    adaptive_keyframes: bool = False,
    min_keyframe_fps: float = DEFAULT_MIN_KEYFRAME_FPS,
    max_keyframe_fps: float = 0.0,
) -> str:
    context = resolve_scene_context(xml_text)
    effects = get_or_create_effects_element(context.video_clip)
    if replace_existing_masks:
        for effect in list(effects):
            if effect.tag == "effect" and effect.attrib.get("id") == MASK_EFFECT_ID:
                effects.remove(effect)

    if not tracks:
        ET.indent(context.root, space=" ")
        return ET.tostring(context.root, encoding="unicode")

    merged_tracks = merge_disjoint_tracks(tracks)

    for index, track in enumerate(merged_tracks):
        effects.append(
            make_mask_effect(
                track,
                context.total_frames,
                frame_width,
                frame_height,
                context.fps,
                shape,
                tilt,
                MASK_OPERATION_WRITE_ON_CLEAR if index == 0 else MASK_OPERATION_MAX,
                keyframe_fps,
                adaptive_keyframes,
                min_keyframe_fps,
                max_keyframe_fps,
            )
        )

    ET.indent(context.root, space=" ")
    return ET.tostring(context.root, encoding="unicode")


def read_text_input(args: argparse.Namespace) -> str:
    if args.clipboard_in:
        return read_from_clipboard()
    if args.input:
        return Path(args.input).read_text(encoding="utf-8")
    if sys.stdin.isatty():
        raise ValueError("No input provided. Pass a file, pipe XML on stdin, or use --clipboard-in.")
    return sys.stdin.read()


def write_text_output(text: str, args: argparse.Namespace) -> None:
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    elif not args.clipboard_out:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")

    if args.clipboard_out:
        write_to_clipboard(text)


def read_from_clipboard() -> str:
    def decode_clipboard_stdout(command_name: str, stdout: bytes) -> str | None:
        encodings = ["utf-8"]
        if command_name == "powershell.exe":
            encodings.extend(["utf-16-le", "utf-16"])
        for encoding in encodings:
            try:
                return stdout.decode(encoding)
            except UnicodeDecodeError:
                continue
        return None

    for command in READ_CLIPBOARD_COMMANDS:
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode == 0:
            decoded = decode_clipboard_stdout(command[0], result.stdout)
            if decoded is not None:
                return decoded
    raise RuntimeError("Unable to read clipboard. Install wl-clipboard or xclip/xsel (Linux), or use pbpaste (macOS). On Windows, ensure PowerShell is available.")


def write_to_clipboard(text: str) -> None:
    for command in WRITE_CLIPBOARD_COMMANDS:
        if shutil.which(command[0]) is None:
            continue
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                start_new_session=True,
            )
        except OSError:
            continue

        try:
            _, stderr = process.communicate(text, timeout=2.0)
        except subprocess.TimeoutExpired:
            # xclip/xsel may keep running to own the clipboard after reading stdin.
            return

        if process.returncode == 0:
            return

        if stderr:
            LOG.debug("Clipboard command %s failed: %s", command[0], stderr.strip())
    raise RuntimeError("Unable to write clipboard. Install wl-clipboard or xclip/xsel (Linux), or use pbcopy (macOS). On Windows, clip.exe should be available by default.")


def generate_tracks(context: SceneContext, args: argparse.Namespace) -> tuple[list[FaceTrack], int, int, list[str]]:
    if not context.source_path.exists():
        raise FileNotFoundError(f"Source media not found: {context.source_path}")

    det_size = parse_det_size(args.det_size)
    detector, providers, cv2 = build_detector(det_size, args.model_name, args.provider_mode)
    LOG.info("Loaded detector providers=%s det_size=%s model=%s", providers, det_size, args.model_name)

    detections = iter_clip_detections(
        context,
        detector=detector,
        cv2=cv2,
        process_width=args.process_width,
        min_score=args.min_score,
        progress_every=args.progress_every,
    )
    tracks = track_detections(
        detections,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        max_gap=args.max_gap,
        min_track_length=args.min_track_length,
        smooth_window=args.smooth_window,
    )
    if context.frame_width is None or context.frame_height is None:
        raise RuntimeError("Unable to determine source video resolution")
    if context.frame_width <= 0 or context.frame_height <= 0:
        raise RuntimeError(
            f"Invalid source video resolution {context.frame_width}x{context.frame_height}; dimensions must be positive"
        )
    return tracks, context.frame_width, context.frame_height, providers


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        validate_args(args)

        if args.doctor:
            doctor_report = build_doctor_report(args)
            output_text = format_doctor_report(doctor_report)
            write_text_output(output_text, args)
            return 0

        xml_text = read_text_input(args)
        xml_path = Path(args.input).expanduser() if args.input and not args.clipboard_in else None
        context = resolve_scene_context(xml_text, xml_path=xml_path)
        tracks, frame_width, frame_height, providers = generate_tracks(context, args)
        LOG.info(
            "Generated %s tracks across %s frames from %s using providers=%s",
            len(tracks),
            context.total_frames,
            context.source_path,
            providers,
        )
        if not tracks:
            LOG.info(
                "No face tracks were generated for %s; %s",
                context.source_path,
                "replacing existing generated masks with none"
                if not args.keep_existing_masks
                else "leaving existing masks unchanged",
            )
        output_text = rewrite_scene_with_tracks(
            xml_text,
            tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            shape=args.shape,
            tilt=args.tilt,
            replace_existing_masks=not args.keep_existing_masks,
            keyframe_fps=args.keyframe_fps,
            adaptive_keyframes=args.adaptive_keyframes,
            min_keyframe_fps=args.min_keyframe_fps,
            max_keyframe_fps=args.max_keyframe_fps,
        )
        write_text_output(output_text, args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:
        LOG.debug("Unexpected internal failure", exc_info=True)
        LOG.error("Unexpected internal failure: %s: %s", exc.__class__.__name__, exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())