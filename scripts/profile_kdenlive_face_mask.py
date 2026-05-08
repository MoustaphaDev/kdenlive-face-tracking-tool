#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import functools
import math
import shlex
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import kdenlive_face_mask as mod


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    cli_args_text: str
    total_s: float
    timings: dict[str, float]
    counts: dict[str, int]
    result: mod.ClipProcessingResult
    effect_count: int
    sample_count: int
    synthetic_sample_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile and compare Kdenlive face-mask runs against real copied-clip XML inputs.",
    )
    parser.add_argument("input", help="Copied-clip XML file to profile.")
    parser.add_argument(
        "--scenario",
        action="append",
        nargs=2,
        metavar=("NAME", "CLI_ARGS"),
        help="Scenario name and a quoted string of CLI arguments to pass to kdenlive-face-mask.",
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU required for a sparse run box to count as matching the baseline.",
    )
    return parser.parse_args(argv)


def sample_counts(tracks: list[mod.FaceTrack]) -> tuple[int, int]:
    total = 0
    synthetic = 0
    for track in tracks:
        total += len(track.samples)
        synthetic += sum(1 for sample in track.samples if sample.synthetic)
    return total, synthetic


def mask_effect_count(output_text: str) -> int:
    root = ET.fromstring(output_text)
    return len(root.findall(f".//effect[@id='{mod.MASK_EFFECT_ID}']"))


@contextlib.contextmanager
def instrument_module():
    timings: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    originals: dict[str, Any] = {}
    patched_cv2: list[tuple[Any, Any, Any]] = []

    def record(name: str, start_time: float) -> None:
        timings[name] += time.perf_counter() - start_time
        counts[name] += 1

    def wrap_function(name: str, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                record(name, start_time)

        return wrapped

    def wrap_generator(name: str, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                yield from fn(*args, **kwargs)
            finally:
                record(name, start_time)

        return wrapped

    class DetectorProxy:
        def __init__(self, inner):
            self._inner = inner

        def get(self, *args, **kwargs):
            start_time = time.perf_counter()
            try:
                return self._inner.get(*args, **kwargs)
            finally:
                record("detector.get", start_time)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    class CaptureProxy:
        def __init__(self, inner):
            self._inner = inner

        def read(self, *args, **kwargs):
            start_time = time.perf_counter()
            try:
                return self._inner.read(*args, **kwargs)
            finally:
                record("capture.read", start_time)

        def grab(self, *args, **kwargs):
            start_time = time.perf_counter()
            try:
                return self._inner.grab(*args, **kwargs)
            finally:
                record("capture.grab", start_time)

        def set(self, *args, **kwargs):
            start_time = time.perf_counter()
            try:
                return self._inner.set(*args, **kwargs)
            finally:
                record("capture.set", start_time)

        def get(self, *args, **kwargs):
            return self._inner.get(*args, **kwargs)

        def isOpened(self, *args, **kwargs):
            return self._inner.isOpened(*args, **kwargs)

        def release(self, *args, **kwargs):
            return self._inner.release(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    def patch_cv2(cv2):
        original_resize = cv2.resize
        original_video_capture = cv2.VideoCapture

        def resize(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return original_resize(*args, **kwargs)
            finally:
                record("cv2.resize", start_time)

        def video_capture(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                capture = original_video_capture(*args, **kwargs)
            finally:
                record("cv2.VideoCapture", start_time)
            return CaptureProxy(capture)

        cv2.resize = resize
        cv2.VideoCapture = video_capture
        patched_cv2.append((cv2, original_resize, original_video_capture))
        return cv2

    originals["build_detector"] = mod.build_detector

    def build_detector(*args, **kwargs):
        start_time = time.perf_counter()
        detector, providers, cv2 = originals["build_detector"](*args, **kwargs)
        record("build_detector", start_time)
        return DetectorProxy(detector), providers, patch_cv2(cv2)

    mod.build_detector = build_detector

    for name in (
        "detect_faces",
        "track_detections",
        "smooth_track",
        "build_mask_frames",
        "thin_mask_frames",
        "build_mask_effect_keyframes",
        "rewrite_scene_with_tracks",
        "generate_tracks",
        "process_clip_xml",
    ):
        originals[name] = getattr(mod, name)
        setattr(mod, name, wrap_function(name, originals[name]))

    originals["iter_clip_detections"] = mod.iter_clip_detections
    mod.iter_clip_detections = wrap_generator("iter_clip_detections", originals["iter_clip_detections"])

    try:
        yield timings, counts
    finally:
        for name, value in originals.items():
            setattr(mod, name, value)
        for cv2, original_resize, original_video_capture in patched_cv2:
            cv2.resize = original_resize
            cv2.VideoCapture = original_video_capture


def build_frame_boxes(tracks: list[mod.FaceTrack]) -> dict[int, list[mod.FaceBox]]:
    boxes_by_frame: defaultdict[int, list[mod.FaceBox]] = defaultdict(list)
    for track in tracks:
        for sample in track.samples:
            boxes_by_frame[sample.frame_index].append(sample.to_box())
    return dict(boxes_by_frame)


def greedy_match_boxes(
    baseline_boxes: list[mod.FaceBox],
    candidate_boxes: list[mod.FaceBox],
) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float, int, int]] = []
    for baseline_index, baseline_box in enumerate(baseline_boxes):
        for candidate_index, candidate_box in enumerate(candidate_boxes):
            iou = mod.box_iou(baseline_box, candidate_box)
            center_error_ratio = math.hypot(
                baseline_box.cx - candidate_box.cx,
                baseline_box.cy - candidate_box.cy,
            ) / max(baseline_box.diagonal, candidate_box.diagonal, 1.0)
            candidates.append((-iou, center_error_ratio, baseline_index, candidate_index))

    matched_baseline: set[int] = set()
    matched_candidate: set[int] = set()
    matches: list[tuple[float, float]] = []
    for neg_iou, center_error_ratio, baseline_index, candidate_index in sorted(candidates):
        if baseline_index in matched_baseline or candidate_index in matched_candidate:
            continue
        matched_baseline.add(baseline_index)
        matched_candidate.add(candidate_index)
        matches.append((-neg_iou, center_error_ratio))
    return matches


def compare_results(
    baseline: ScenarioResult,
    candidate: ScenarioResult,
    *,
    iou_threshold: float,
) -> dict[str, float]:
    baseline_boxes = build_frame_boxes(baseline.result.tracks)
    candidate_boxes = build_frame_boxes(candidate.result.tracks)
    frame_indices = sorted(set(baseline_boxes) | set(candidate_boxes))

    total_baseline_boxes = 0
    total_candidate_boxes = 0
    matched_ious: list[float] = []
    matched_center_errors: list[float] = []
    for frame_index in frame_indices:
        baseline_frame_boxes = baseline_boxes.get(frame_index, [])
        candidate_frame_boxes = candidate_boxes.get(frame_index, [])
        total_baseline_boxes += len(baseline_frame_boxes)
        total_candidate_boxes += len(candidate_frame_boxes)
        for iou, center_error in greedy_match_boxes(baseline_frame_boxes, candidate_frame_boxes):
            if iou < iou_threshold:
                continue
            matched_ious.append(iou)
            matched_center_errors.append(center_error)

    matched_count = len(matched_ious)
    return {
        "speedup_vs_baseline": baseline.total_s / max(candidate.total_s, 1e-9),
        "baseline_box_coverage": matched_count / max(total_baseline_boxes, 1),
        "candidate_box_precision": matched_count / max(total_candidate_boxes, 1),
        "mean_iou": sum(matched_ious) / max(matched_count, 1),
        "median_iou": median(matched_ious) if matched_ious else 0.0,
        "mean_center_error_ratio": sum(matched_center_errors) / max(matched_count, 1),
        "matched_boxes": float(matched_count),
        "baseline_boxes": float(total_baseline_boxes),
        "candidate_boxes": float(total_candidate_boxes),
    }


def run_scenario(input_path: Path, name: str, cli_args_text: str) -> ScenarioResult:
    scenario_argv = [str(input_path), *shlex.split(cli_args_text)]
    scenario_args = mod.parse_args(scenario_argv)
    mod.validate_args(scenario_args)
    xml_text = input_path.read_text(encoding="utf-8")

    with instrument_module() as (timings, counts):
        start_time = time.perf_counter()
        result = mod.process_clip_xml(xml_text, scenario_args, xml_path=input_path)
        total_s = time.perf_counter() - start_time

    sample_count, synthetic_sample_count = sample_counts(result.tracks)
    return ScenarioResult(
        name=name,
        cli_args_text=cli_args_text,
        total_s=total_s,
        timings=dict(timings),
        counts=dict(counts),
        result=result,
        effect_count=mask_effect_count(result.output_text),
        sample_count=sample_count,
        synthetic_sample_count=synthetic_sample_count,
    )


def format_seconds(value: float) -> str:
    return f"{value:.4f}s"


def print_scenario(result: ScenarioResult) -> None:
    synthetic_ratio = result.synthetic_sample_count / max(result.sample_count, 1)
    print(f"Scenario {result.name}")
    print(f"  args: {result.cli_args_text or '(default)'}")
    print(f"  total: {format_seconds(result.total_s)}")
    print(
        f"  tracks: {len(result.result.tracks)} | effects: {result.effect_count} | "
        f"samples: {result.sample_count} | synthetic_ratio: {synthetic_ratio:.2%}"
    )
    for key in (
        "build_detector",
        "detector.get",
        "cv2.resize",
        "capture.read",
        "track_detections",
        "smooth_track",
        "rewrite_scene_with_tracks",
    ):
        if key not in result.timings:
            continue
        print(f"  {key}: {format_seconds(result.timings[key])} count={result.counts.get(key, 0)}")


def print_comparison(baseline: ScenarioResult, candidate: ScenarioResult, *, iou_threshold: float) -> None:
    comparison = compare_results(baseline, candidate, iou_threshold=iou_threshold)
    print(f"Comparison {candidate.name} vs {baseline.name}")
    print(f"  speedup: {comparison['speedup_vs_baseline']:.2f}x")
    print(
        f"  matched baseline boxes at IoU>={iou_threshold:.2f}: "
        f"{comparison['baseline_box_coverage']:.2%}"
    )
    print(
        f"  candidate box precision at IoU>={iou_threshold:.2f}: "
        f"{comparison['candidate_box_precision']:.2%}"
    )
    print(f"  mean IoU of matches: {comparison['mean_iou']:.4f}")
    print(f"  median IoU of matches: {comparison['median_iou']:.4f}")
    print(f"  mean normalized center error: {comparison['mean_center_error_ratio']:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    scenarios = args.scenario or [("baseline", "")]
    mod.configure_logging("ERROR")

    results = [run_scenario(input_path, name, cli_args_text) for name, cli_args_text in scenarios]
    for result in results:
        print_scenario(result)

    if len(results) > 1:
        baseline = results[0]
        for candidate in results[1:]:
            print_comparison(baseline, candidate, iou_threshold=args.match_iou_threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())