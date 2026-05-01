#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

LOG = logging.getLogger("kdenlive-face-mask")

MASK_EFFECT_ID = "mask_start-frei0r.alphaspot"
DEFAULT_SHAPE = 0.38
DEFAULT_TILT = 0.5
DEFAULT_PROGRESS_EVERY = 120
MODERATE_POSITION_MOTION_RATIO = 0.06
FAST_POSITION_MOTION_RATIO = 0.12


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
        choices=("auto", "rocm", "migraphx", "cpu"),
        help="ONNX Runtime provider preference. auto tries ROCm first, then CPU.",
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
    return parser.parse_args(argv)


def configure_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, level_name), format="[maskgen] %(levelname)s: %(message)s")


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


def resolve_source_path(root: ET.Element, resource: str) -> Path:
    path = Path(resource).expanduser()
    if path.is_absolute():
        return path
    scene_root = root.attrib.get("root")
    if scene_root:
        return (Path(scene_root).expanduser() / path).resolve()
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


def resolve_scene_context(xml_text: str) -> SceneContext:
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
        source_path=resolve_source_path(root, resource),
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
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to read the source media") from exc

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required to warm up InsightFace") from exc

    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to run InsightFace") from exc

    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except ImportError as exc:
        raise RuntimeError("insightface is required to run face detection") from exc

    return cv2, np, ort, FaceAnalysis


def build_detector(det_size: tuple[int, int], model_name: str, provider_mode: str):
    cv2, np, ort, FaceAnalysis = _import_detection_modules()
    available = ort.get_available_providers()
    provider_sets: list[list[str]] = []

    if provider_mode == "auto":
        if "ROCMExecutionProvider" in available:
            provider_sets.append(["ROCMExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "rocm":
        if "ROCMExecutionProvider" in available:
            provider_sets.append(["ROCMExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "migraphx":
        if "MIGraphXExecutionProvider" in available:
            provider_sets.append(["MIGraphXExecutionProvider", "CPUExecutionProvider"])
        provider_sets.append(["CPUExecutionProvider"])
    elif provider_mode == "cpu":
        provider_sets.append(["CPUExecutionProvider"])
    else:
        raise ValueError(f"Unknown provider-mode {provider_mode!r}")

    last_error: Exception | None = None
    for providers in provider_sets:
        try:
            app = FaceAnalysis(name=model_name, allowed_modules=["detection"], providers=providers)
            app.prepare(ctx_id=0, det_size=det_size)
            app.get(np.zeros((360, 640, 3), dtype=np.uint8))
            return app, providers, cv2
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


def build_mask_frames(track: FaceTrack, total_frames: int, frame_width: int, frame_height: int) -> list[MaskFrame]:
    sample_by_frame = {sample.frame_index: sample for sample in track.samples}
    sorted_frame_indices = sorted(sample_by_frame)

    def normalized(sample: TrackSample, frame_index: int | None = None, zero_size: bool = False) -> MaskFrame:
        pos_x = min(max(sample.cx / frame_width, 0.0), 1.0)
        pos_y = min(max(sample.cy / frame_height, 0.0), 1.0)
        size_x = 0.0 if zero_size else min(max(sample.half_w / frame_width, 0.0), 1.0)
        size_y = 0.0 if zero_size else min(max(sample.half_h / frame_height, 0.0), 1.0)
        return MaskFrame(sample.frame_index if frame_index is None else frame_index, pos_x, pos_y, size_x, size_y)

    segments: list[list[int]] = [[sorted_frame_indices[0]]]
    for frame_index in sorted_frame_indices[1:]:
        if frame_index == segments[-1][-1] + 1:
            segments[-1].append(frame_index)
        else:
            segments.append([frame_index])

    frames: dict[int, MaskFrame] = {}
    first_segment_start = segments[0][0]
    if first_segment_start > 0:
        frames[0] = normalized(sample_by_frame[first_segment_start], frame_index=0, zero_size=True)

    for segment in segments:
        segment_start = segment[0]
        segment_end = segment[-1]
        first = sample_by_frame[segment_start]
        last = sample_by_frame[segment_end]

        if segment_start == 0:
            frames[0] = normalized(first, frame_index=0, zero_size=True)
        else:
            frames[segment_start - 1] = normalized(first, frame_index=segment_start - 1, zero_size=True)

        for frame_index in segment:
            if segment_start == 0 and frame_index == 0:
                continue
            frames[frame_index] = normalized(sample_by_frame[frame_index])

        if segment_start == 0 and len(segment) == 1 and total_frames > 1:
            frames[1] = normalized(first, frame_index=1)

        if segment_end < total_frames - 1:
            frames[segment_end + 1] = normalized(last, frame_index=segment_end + 1, zero_size=True)

    return [frames[index] for index in sorted(frames)]


def merge_disjoint_tracks(tracks: Sequence[FaceTrack]) -> list[FaceTrack]:
    if not tracks:
        return []

    merged_tracks: list[FaceTrack] = []
    sorted_tracks = sorted(tracks, key=lambda track: (track.first_frame(), track.last_frame(), track.average_cx(), track.track_id))
    for track in sorted_tracks:
        if not merged_tracks:
            merged_tracks.append(FaceTrack(track_id=track.track_id, samples=list(track.samples)))
            continue

        previous = merged_tracks[-1]
        if track.first_frame() > previous.last_frame():
            previous.samples.extend(track.samples)
            continue

        merged_tracks.append(FaceTrack(track_id=track.track_id, samples=list(track.samples)))

    return merged_tracks


def build_keyframe_string(frames: Sequence[MaskFrame], selector) -> str:
    return ";".join(f"{frame.frame_index}={format_float(selector(frame))}" for frame in frames)


def build_constant_keyframe_string(frames: Sequence[MaskFrame], value: float) -> str:
    return ";".join(f"{frame.frame_index}={format_float(value)}" for frame in frames)


def make_property(name: str, value: str) -> ET.Element:
    prop = ET.Element("property", {"name": name})
    prop.text = value
    return prop


def make_mask_effect(track: FaceTrack, total_frames: int, frame_width: int, frame_height: int, shape: float, tilt: float) -> ET.Element:
    effect = ET.Element("effect", {"id": MASK_EFFECT_ID})
    frames = build_mask_frames(track, total_frames, frame_width, frame_height)
    effect.append(make_property("kdenlive:collapsed", "1"))
    effect.append(make_property("filter.Min", build_constant_keyframe_string(frames, 0.0)))
    effect.append(make_property("filter.Transition width", build_constant_keyframe_string(frames, 0.0)))
    effect.append(make_property("filter.Tilt", build_constant_keyframe_string(frames, tilt)))
    effect.append(make_property("filter.Size Y", build_keyframe_string(frames, lambda item: item.size_y)))
    effect.append(make_property("filter.Max", build_constant_keyframe_string(frames, 1.0)))
    effect.append(make_property("filter.Size X", build_keyframe_string(frames, lambda item: item.size_x)))
    effect.append(make_property("filter.Position Y", build_keyframe_string(frames, lambda item: item.pos_y)))
    effect.append(make_property("filter.Position X", build_keyframe_string(frames, lambda item: item.pos_x)))
    effect.append(make_property("filter.Shape", format_float(shape)))
    effect.append(make_property("filter.Operation", "0"))
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
) -> str:
    if not tracks:
        raise ValueError("No face tracks were generated for this clip")

    context = resolve_scene_context(xml_text)
    effects = get_or_create_effects_element(context.video_clip)
    merged_tracks = merge_disjoint_tracks(tracks)
    if replace_existing_masks:
        for effect in list(effects):
            if effect.tag == "effect" and effect.attrib.get("id") == MASK_EFFECT_ID:
                effects.remove(effect)

    for track in merged_tracks:
        effects.append(make_mask_effect(track, context.total_frames, frame_width, frame_height, shape, tilt))

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
    commands = [
        ["wl-paste", "--no-newline", "--type", "text/plain;charset=utf-8"],
        ["wl-paste", "--no-newline", "--type", "text/plain"],
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-out"],
        ["xsel", "--clipboard", "--output"],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode == 0:
            try:
                return result.stdout.decode("utf-8")
            except UnicodeDecodeError:
                continue
    raise RuntimeError("Unable to read clipboard. Install wl-clipboard, xclip, or xsel.")


def write_to_clipboard(text: str) -> None:
    commands = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard", "-in"],
        ["xsel", "--clipboard", "--input"],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
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
    raise RuntimeError("Unable to write clipboard. Install wl-clipboard, xclip, or xsel.")


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
    return tracks, context.frame_width, context.frame_height, providers


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        xml_text = read_text_input(args)
        context = resolve_scene_context(xml_text)
        tracks, frame_width, frame_height, providers = generate_tracks(context, args)
        LOG.info(
            "Generated %s tracks across %s frames from %s using providers=%s",
            len(tracks),
            context.total_frames,
            context.source_path,
            providers,
        )
        output_text = rewrite_scene_with_tracks(
            xml_text,
            tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            shape=args.shape,
            tilt=args.tilt,
            replace_existing_masks=not args.keep_existing_masks,
        )
        write_text_output(output_text, args)
    except Exception as exc:
        LOG.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())