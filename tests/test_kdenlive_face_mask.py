import contextlib
import io
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from tools.kdenlive_face_mask import (
    MASK_EFFECT_ID,
    FaceTrack,
    MaskFrame,
    TrackSample,
    build_detector,
    map_clip_frame_to_source_frame,
    resolve_scene_context,
    rewrite_scene_with_tracks,
    smooth_track,
    thin_mask_frames,
)


SCENE_XML = """<kdenlive-scene audioTracks="2" documentid="1777564221124" duration="29" fps="60" mainClip="15;14" masterTrack="2" offset="0" videoTracks="2">
 <clip audioStream="1" audioTrack="1" binid="4" id="14" in="0" mirrorTrack="2" out="29" playlist="0" position="0" speed="1.000000" state="2" track="1">
  <effects parentIn="0"/>
 </clip>
 <clip audioStream="1" binid="4" id="15" in="0" out="29" playlist="0" position="0" speed="1.000000" state="1" track="2">
  <effects parentIn="0">
   <effect id="mask_start-frei0r.alphaspot">
    <property name="kdenlive:collapsed">1</property>
    <property name="filter.Min">0=0;2=0</property>
    <property name="filter.Transition width">0=0;2=0</property>
    <property name="filter.Tilt">0=0.5;2=0.5</property>
    <property name="filter.Size Y">0=0.04;2=0.04</property>
    <property name="filter.Max">0=1;2=1</property>
    <property name="filter.Size X">0=0.02;2=0.02</property>
    <property name="filter.Position Y">0=0.1;2=0.1</property>
    <property name="filter.Position X">0=0.2;2=0.2</property>
    <property name="filter.Shape">0.38</property>
    <property name="filter.Operation">0</property>
   </effect>
  </effects>
 </clip>
 <bin>
  <chain id="chain0" out="29" type="3">
   <property name="length">30</property>
   <property name="eof">pause</property>
   <property name="resource">/tmp/sample.mp4</property>
   <property name="mlt_service">avformat</property>
   <property name="seekable">1</property>
   <property name="format">3</property>
   <property name="audio_index">1</property>
   <property name="video_index">0</property>
   <property name="vstream">0</property>
   <property name="astream">0</property>
   <property name="kdenlive:id">4</property>
   <property name="meta.media.width">1920</property>
   <property name="meta.media.height">1080</property>
  </chain>
 </bin>
 <groups>[]</groups>
</kdenlive-scene>
"""

NO_EFFECT_SCENE_XML = """<kdenlive-scene audioTracks="2" documentid="1777564221124" duration="11140" fps="60" mainClip="26;27" masterAudioTrack="1" masterTrack="2" offset="13" videoTracks="2">
 <clip audioStream="1" binid="4" id="26" in="0" out="11139" playlist="0" position="13" speed="1.000000" state="1" track="2">
  <effects parentIn="0"/>
 </clip>
 <clip audioStream="1" audioTrack="1" binid="4" id="27" in="0" mirrorTrack="2" out="11139" playlist="0" position="13" speed="1.000000" state="2" track="1">
  <effects parentIn="0"/>
 </clip>
 <bin>
  <chain id="chain0" out="11139" type="3">
   <property name="length">11140</property>
   <property name="eof">pause</property>
   <property name="resource">/tmp/sample.mp4</property>
   <property name="mlt_service">avformat</property>
   <property name="seekable">1</property>
   <property name="format">3</property>
   <property name="audio_index">1</property>
   <property name="video_index">0</property>
   <property name="vstream">0</property>
   <property name="astream">0</property>
   <property name="kdenlive:control_uuid">{80710ad8-8f60-4814-bd7c-1ce96f04ef62}</property>
   <property name="kdenlive:id">4</property>
   <property name="kdenlive:clip_type">0</property>
   <property name="kdenlive:file_size">143195469</property>
   <property name="kdenlive:file_hash">8d7ea321ff3002effda82789ed731265</property>
   <property name="kdenlive:folderid">-1</property>
   <property name="meta.media.width">1920</property>
   <property name="meta.media.height">1080</property>
  </chain>
 </bin>
 <groups>[
    {
        "children": [
            {
                "data": "2:13:-1",
                "leaf": "clip",
                "type": "Leaf"
            },
            {
                "data": "1:13:-1",
                "leaf": "clip",
                "type": "Leaf"
            }
        ],
        "type": "AVSplit"
    }
]
</groups>
</kdenlive-scene>
"""


def make_track(track_id: int, frames: range, base_x: float) -> FaceTrack:
    return FaceTrack(
        track_id=track_id,
        samples=[
            TrackSample(
                frame_index=frame,
                cx=base_x + frame * 3.0,
                cy=120.0 + frame * 1.5,
                half_w=38.0,
                half_h=54.0,
            )
            for frame in frames
        ],
    )


class ResolveSceneContextTests(unittest.TestCase):
    def test_extracts_resource_and_frame_range(self) -> None:
        context = resolve_scene_context(SCENE_XML)

        self.assertEqual(context.source_path.as_posix(), "/tmp/sample.mp4")
        self.assertEqual(context.fps, 60.0)
        self.assertEqual(context.in_frame, 0)
        self.assertEqual(context.out_frame, 29)
        self.assertEqual(context.total_frames, 30)
        self.assertEqual(context.frame_width, 1920)
        self.assertEqual(context.frame_height, 1080)


class SourceFrameMappingTests(unittest.TestCase):
    def test_identity_mapping_when_clip_and_source_fps_match(self) -> None:
        self.assertEqual(123, map_clip_frame_to_source_frame(123, 60.0, 60.0, 500))

    def test_mixed_fps_mapping_clamps_to_last_source_frame(self) -> None:
        self.assertEqual(0, map_clip_frame_to_source_frame(0, 60.0, 29.97, 436))
        self.assertEqual(218, map_clip_frame_to_source_frame(437, 60.0, 29.97, 436))
        self.assertEqual(436, map_clip_frame_to_source_frame(875, 60.0, 29.97, 436))


class DetectorInitializationTests(unittest.TestCase):
    def test_build_detector_redirects_third_party_stdout_to_stderr(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["ROCMExecutionProvider", "CPUExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                print(f"Applied providers: {providers}")
                self.name = name
                self.allowed_modules = allowed_modules
                self.providers = providers

            def prepare(self, *, ctx_id, det_size):
                print(f"set det-size: {det_size}")
                self.ctx_id = ctx_id
                self.det_size = det_size

            def get(self, _frame):
                print("model ignore: fake-model")
                return []

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "tools.kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                detector, providers, cv2 = build_detector((320, 320), "buffalo_s", "rocm")

        self.assertEqual("", stdout.getvalue())
        self.assertIn("Applied providers:", stderr.getvalue())
        self.assertIn("set det-size:", stderr.getvalue())
        self.assertIn("model ignore:", stderr.getvalue())
        self.assertEqual(["ROCMExecutionProvider", "CPUExecutionProvider"], providers)
        self.assertEqual("fake-cv2", cv2)
        self.assertIsInstance(detector, FakeFaceAnalysis)


class RewriteSceneWithTracksTests(unittest.TestCase):
    def test_replaces_existing_mask_effects(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(1, range(0, 6), 300.0), make_track(2, range(8, 12), 700.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        video_clip = root.find("./clip[@id='15']")
        self.assertIsNotNone(video_clip)
        effects = video_clip.findall("./effects/effect")
        self.assertEqual(1, len(effects))
        self.assertTrue(all(effect.attrib.get("id") == MASK_EFFECT_ID for effect in effects))

    def test_overlapping_tracks_remain_separate_effects(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(1, range(0, 6), 300.0), make_track(2, range(5, 9), 700.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effects = root.findall("./clip[@id='15']/effects/effect")
        self.assertEqual(2, len(effects))

        operations = [effect.find("./property[@name='filter.Operation']").text for effect in effects]
        self.assertEqual(["0", "0.3"], operations)

    def test_zero_sizes_are_inserted_outside_track_span(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(7, range(10, 13), 520.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effect = root.find("./clip[@id='15']/effects/effect")
        self.assertIsNotNone(effect)
        size_x = effect.find("./property[@name='filter.Size X']")
        self.assertIsNotNone(size_x)
        self.assertEqual("9=0;10=0.0197917;11=0.0197917;12=0.0197917;13=0", size_x.text)

    def test_mask_keyframes_stay_local_to_track_span(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(1, range(0, 6), 300.0), make_track(2, range(8, 12), 700.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effect = root.find("./clip[@id='15']/effects/effect")
        self.assertIsNotNone(effect)

        size_x = effect.find("./property[@name='filter.Size X']")
        self.assertIsNotNone(size_x)
        self.assertEqual(
            "0=0;1=0.0197917;2=0.0197917;3=0.0197917;4=0.0197917;5=0.0197917;6=0;7=0;8=0.0197917;9=0.0197917;10=0.0197917;11=0.0197917;12=0",
            size_x.text,
        )

    def test_non_zero_start_track_uses_pre_segment_zero_anchor_only(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(11, range(8, 12), 700.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effect = root.find("./clip[@id='15']/effects/effect")
        self.assertIsNotNone(effect)
        size_x = effect.find("./property[@name='filter.Size X']")
        self.assertIsNotNone(size_x)
        self.assertEqual("7=0;8=0.0197917;9=0.0197917;10=0.0197917;11=0.0197917;12=0", size_x.text)

    def test_start_of_clip_track_begins_with_zero_size_keyframe(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(9, range(0, 3), 420.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effect = root.find("./clip[@id='15']/effects/effect")
        self.assertIsNotNone(effect)
        size_x = effect.find("./property[@name='filter.Size X']")
        self.assertIsNotNone(size_x)
        self.assertEqual("0=0;1=0.0197917;2=0.0197917;3=0", size_x.text)

    def test_adds_mask_to_empty_video_clip_effects(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            NO_EFFECT_SCENE_XML,
            [make_track(3, range(0, 5), 400.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        video_clip = root.find("./clip[@id='26']")
        audio_clip = root.find("./clip[@id='27']")
        self.assertIsNotNone(video_clip)
        self.assertIsNotNone(audio_clip)
        self.assertEqual(1, len(video_clip.findall("./effects/effect")))
        self.assertEqual(0, len(audio_clip.findall("./effects/effect")))
        self.assertEqual(MASK_EFFECT_ID, video_clip.find("./effects/effect").attrib.get("id"))

    def test_fixed_keyframe_fps_reduces_serialized_keyframes(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(12, range(0, 8), 420.0)],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
            keyframe_fps=15.0,
        )

        root = ET.fromstring(rewritten)
        effect = root.find("./clip[@id='15']/effects/effect")
        self.assertIsNotNone(effect)
        size_x = effect.find("./property[@name='filter.Size X']")
        self.assertIsNotNone(size_x)
        self.assertEqual("0=0;1=0.0197917;5=0.0197917;7=0.0197917;8=0", size_x.text)


class KeyframeDensityTests(unittest.TestCase):
    @staticmethod
    def make_frames(indices: range, *, start_zero: bool = True, end_zero_frame: int | None = None, step: float = 0.002) -> list[MaskFrame]:
        frames: list[MaskFrame] = []
        if start_zero:
            frames.append(MaskFrame(0, 0.5, 0.5, 0.0, 0.0))
        for offset, frame_index in enumerate(indices, start=1):
            frames.append(MaskFrame(frame_index, 0.5 + offset * step, 0.4, 0.02, 0.04))
        if end_zero_frame is not None:
            frames.append(MaskFrame(end_zero_frame, 0.5 + (len(indices) + 1) * step, 0.4, 0.0, 0.0))
        return frames

    def test_fixed_keyframe_fps_preserves_zero_size_anchors(self) -> None:
        frames = self.make_frames(range(1, 6), end_zero_frame=6)

        thinned = thin_mask_frames(
            frames,
            clip_fps=60.0,
            keyframe_fps=15.0,
            adaptive_keyframes=False,
            min_keyframe_fps=12.0,
            max_keyframe_fps=0.0,
        )

        self.assertEqual([0, 1, 5, 6], [frame.frame_index for frame in thinned])

    def test_adaptive_keyframes_keep_fast_motion_denser_than_slow_motion(self) -> None:
        slow_frames = self.make_frames(range(1, 7), end_zero_frame=7, step=0.0002)
        fast_frames = self.make_frames(range(1, 7), end_zero_frame=7, step=0.02)

        slow = thin_mask_frames(
            slow_frames,
            clip_fps=60.0,
            keyframe_fps=0.0,
            adaptive_keyframes=True,
            min_keyframe_fps=15.0,
            max_keyframe_fps=60.0,
        )
        fast = thin_mask_frames(
            fast_frames,
            clip_fps=60.0,
            keyframe_fps=0.0,
            adaptive_keyframes=True,
            min_keyframe_fps=15.0,
            max_keyframe_fps=60.0,
        )

        slow_visible_count = sum(1 for frame in slow if frame.size_x > 0.0)
        fast_visible_count = sum(1 for frame in fast if frame.size_x > 0.0)

        self.assertLess(slow_visible_count, fast_visible_count)
        self.assertEqual([0, 7], [frame.frame_index for frame in slow if frame.size_x == 0.0])
        self.assertEqual([0, 7], [frame.frame_index for frame in fast if frame.size_x == 0.0])


class SmoothTrackTests(unittest.TestCase):
    def test_preserves_sharp_position_changes(self) -> None:
        track = FaceTrack(
            track_id=1,
            samples=[
                TrackSample(frame_index=0, cx=100.0, cy=100.0, half_w=20.0, half_h=20.0),
                TrackSample(frame_index=1, cx=100.0, cy=100.0, half_w=20.0, half_h=20.0),
                TrackSample(frame_index=2, cx=180.0, cy=100.0, half_w=20.0, half_h=20.0),
                TrackSample(frame_index=3, cx=180.0, cy=100.0, half_w=20.0, half_h=20.0),
                TrackSample(frame_index=4, cx=180.0, cy=100.0, half_w=20.0, half_h=20.0),
            ],
        )

        smooth_track(track, radius=2)

        self.assertEqual(100.0, track.samples[1].cx)
        self.assertEqual(180.0, track.samples[2].cx)
        self.assertEqual(180.0, track.samples[3].cx)


if __name__ == "__main__":
    unittest.main()