import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import builtins
from typing import Sequence
import unittest
import unittest.mock
import xml.etree.ElementTree as ET
from unittest.mock import patch
from xml.sax.saxutils import escape

from kdenlive_face_mask import (
    MASK_EFFECT_ID,
    FaceTrack,
    MaskFrame,
    TrackSample,
    build_mask_frames,
    build_detector,
    detect_onnxruntime_support,
    main,
    map_clip_frame_to_source_frame,
    probe_provider_mode,
    read_from_clipboard,
    resolve_scene_context,
    rewrite_scene_with_tracks,
    smooth_track,
    thin_mask_frames,
    write_to_clipboard,
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


def make_smoke_scene_xml(resource_path: str, *, frame_count: int, frame_width: int, frame_height: int, fps: float) -> str:
    out_frame = frame_count - 1
    return f"""<kdenlive-scene audioTracks=\"0\" documentid=\"1\" duration=\"{out_frame}\" fps=\"{fps}\" mainClip=\"26\" masterTrack=\"1\" offset=\"0\" videoTracks=\"1\">
 <clip binid=\"4\" id=\"26\" in=\"0\" out=\"{out_frame}\" playlist=\"0\" position=\"0\" speed=\"1.000000\" state=\"1\" track=\"1\">
  <effects parentIn=\"0\"/>
 </clip>
 <bin>
  <chain id=\"chain0\" out=\"{out_frame}\" type=\"3\">
   <property name=\"length\">{frame_count}</property>
   <property name=\"eof\">pause</property>
   <property name=\"resource\">{escape(resource_path)}</property>
   <property name=\"mlt_service\">avformat</property>
   <property name=\"seekable\">1</property>
   <property name=\"format\">3</property>
   <property name=\"video_index\">0</property>
   <property name=\"vstream\">0</property>
   <property name=\"kdenlive:id\">4</property>
   <property name=\"meta.media.width\">{frame_width}</property>
   <property name=\"meta.media.height\">{frame_height}</property>
  </chain>
 </bin>
 <groups>[]</groups>
</kdenlive-scene>
"""


def fake_provider_model(providers: Sequence[str]):
    return type(
        "FakeModel",
        (),
        {
            "session": type(
                "FakeSession",
                (),
                {"get_providers": lambda self: list(providers)},
            )(),
        },
    )()


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

    def test_resolves_relative_resource_against_input_xml_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "project"
            project_dir.mkdir()
            xml_path = project_dir / "copied-clip.xml"
            xml_text = make_smoke_scene_xml(
                "media/sample.mp4",
                frame_count=5,
                frame_width=64,
                frame_height=48,
                fps=30.0,
            )
            xml_path.write_text(xml_text, encoding="utf-8")

            context = resolve_scene_context(xml_text, xml_path=xml_path)

        self.assertEqual(project_dir / "media" / "sample.mp4", context.source_path)


class SourceFrameMappingTests(unittest.TestCase):
    def test_identity_mapping_when_clip_and_source_fps_match(self) -> None:
        self.assertEqual(123, map_clip_frame_to_source_frame(123, 60.0, 60.0, 500))

    def test_mixed_fps_mapping_clamps_to_last_source_frame(self) -> None:
        self.assertEqual(0, map_clip_frame_to_source_frame(0, 60.0, 29.97, 436))
        self.assertEqual(218, map_clip_frame_to_source_frame(437, 60.0, 29.97, 436))
        self.assertEqual(436, map_clip_frame_to_source_frame(875, 60.0, 29.97, 436))


class RuntimeDependencySmokeTests(unittest.TestCase):
    def test_installed_onnxruntime_exposes_expected_api(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            self.skipTest("onnxruntime is not installed in this environment")

        self.assertTrue(hasattr(ort, "InferenceSession"))
        self.assertTrue(hasattr(ort, "get_available_providers"))


class RuntimeDependencyFailureTests(unittest.TestCase):
    def test_detect_onnxruntime_support_reports_native_loader_failures(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise OSError("missing libonnxruntime.so")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            support = detect_onnxruntime_support()

        self.assertFalse(support["import_ok"])
        self.assertIn("OSError", support["error"])
        self.assertIn("missing libonnxruntime.so", support["error"])


class DetectorInitializationTests(unittest.TestCase):
    def test_build_detector_prefers_cuda_in_auto_mode(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider", "ROCMExecutionProvider", "CUDAExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                self.name = name
                self.allowed_modules = allowed_modules
                self.models = {"detection": fake_provider_model(providers)}

            def prepare(self, *, ctx_id, det_size):
                self.ctx_id = ctx_id
                self.det_size = det_size

            def get(self, _frame):
                return []

        with patch(
            "kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            detector, providers, cv2 = build_detector((320, 320), "buffalo_s", "auto")

        self.assertEqual(["CUDAExecutionProvider", "CPUExecutionProvider"], providers)
        self.assertEqual(["CUDAExecutionProvider", "CPUExecutionProvider"], detector.models["detection"].session.get_providers())
        self.assertEqual("fake-cv2", cv2)

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
                self.models = {"detection": fake_provider_model(providers)}

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
            "kdenlive_face_mask._import_detection_modules",
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

    def test_build_detector_uses_cuda_mode_when_available(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                self.models = {"detection": fake_provider_model(providers)}

            def prepare(self, *, ctx_id, det_size):
                self.ctx_id = ctx_id
                self.det_size = det_size

            def get(self, _frame):
                return []

        with patch(
            "kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            detector, providers, cv2 = build_detector((320, 320), "buffalo_s", "cuda")

        self.assertEqual(["CUDAExecutionProvider", "CPUExecutionProvider"], providers)
        self.assertEqual(["CUDAExecutionProvider", "CPUExecutionProvider"], detector.models["detection"].session.get_providers())
        self.assertEqual("fake-cv2", cv2)

    def test_build_detector_prefers_coreml_in_auto_mode_on_macos(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider", "CoreMLExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                self.models = {"detection": fake_provider_model(providers)}

            def prepare(self, *, ctx_id, det_size):
                pass

            def get(self, _frame):
                return []

        with patch(
            "kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            detector, providers, cv2 = build_detector((320, 320), "buffalo_s", "auto")

        self.assertEqual(["CoreMLExecutionProvider", "CPUExecutionProvider"], providers)

    def test_build_detector_uses_coreml_mode_when_available(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                self.models = {"detection": fake_provider_model(providers)}

            def prepare(self, *, ctx_id, det_size):
                pass

            def get(self, _frame):
                return []

        with patch(
            "kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            detector, providers, cv2 = build_detector((320, 320), "buffalo_s", "coreml")

        self.assertEqual(["CoreMLExecutionProvider", "CPUExecutionProvider"], providers)
        self.assertEqual(["CoreMLExecutionProvider", "CPUExecutionProvider"], detector.models["detection"].session.get_providers())

    def test_build_detector_uses_openvino_mode_when_available(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                self.models = {"detection": fake_provider_model(providers)}

            def prepare(self, *, ctx_id, det_size):
                pass

            def get(self, _frame):
                return []

        with patch(
            "kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            detector, providers, cv2 = build_detector((320, 320), "buffalo_s", "openvino")

        self.assertEqual(["OpenVINOExecutionProvider", "CPUExecutionProvider"], providers)
        self.assertEqual(["OpenVINOExecutionProvider", "CPUExecutionProvider"], detector.models["detection"].session.get_providers())

    def test_build_detector_falls_back_to_cpu_when_coreml_unavailable(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                self.models = {"detection": fake_provider_model(providers)}

            def prepare(self, *, ctx_id, det_size):
                pass

            def get(self, _frame):
                return []

        with patch(
            "kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            detector, providers, cv2 = build_detector((320, 320), "buffalo_s", "coreml")

        self.assertEqual(["CPUExecutionProvider"], providers)

    def test_probe_provider_mode_detects_runtime_fallback_to_cpu(self) -> None:
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        class FakeNP:
            uint8 = object()

            @staticmethod
            def zeros(_shape, dtype=None):
                return {"dtype": dtype}

        class FakeFaceAnalysis:
            def __init__(self, *, name, allowed_modules, providers):
                self.models = {"detection": fake_provider_model(["CPUExecutionProvider"])}

            def prepare(self, *, ctx_id, det_size):
                pass

            def get(self, _frame):
                return []

        with patch(
            "kdenlive_face_mask._import_detection_modules",
            return_value=("fake-cv2", FakeNP(), FakeOrt(), FakeFaceAnalysis),
        ):
            probe = probe_provider_mode((320, 320), "buffalo_s", "cuda")

        self.assertEqual("fallback-to-cpu", probe["status"])
        self.assertEqual(["CPUExecutionProvider"], probe["active_providers"])


class ClipboardTests(unittest.TestCase):
    def test_read_from_clipboard_uses_pbpaste_on_macos(self) -> None:
        def fake_which(cmd: str):
            return "/usr/bin/pbpaste" if cmd == "pbpaste" else None

        with patch("kdenlive_face_mask.shutil.which", side_effect=fake_which):
            with patch(
                "kdenlive_face_mask.subprocess.run",
                return_value=type("R", (), {"returncode": 0, "stdout": b"<xml/>"})()
            ):
                result = read_from_clipboard()

        self.assertEqual("<xml/>", result)

    def test_read_from_clipboard_uses_powershell_on_windows(self) -> None:
        def fake_which(cmd: str):
            return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" if cmd == "powershell.exe" else None

        with patch("kdenlive_face_mask.shutil.which", side_effect=fake_which):
            with patch(
                "kdenlive_face_mask.subprocess.run",
                return_value=type("R", (), {"returncode": 0, "stdout": b"<xml/>"})()
            ) as mock_run:
                result = read_from_clipboard()

        self.assertEqual("<xml/>", result)
        called_cmd = mock_run.call_args[0][0]
        self.assertEqual("powershell.exe", called_cmd[0])
        self.assertIn("Get-Clipboard", called_cmd[-1])

    def test_read_from_clipboard_decodes_utf8_powershell_output(self) -> None:
        def fake_which(cmd: str):
            return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" if cmd == "powershell.exe" else None

        expected = "<clip title=\"cafe 日本\"/>"
        with patch("kdenlive_face_mask.shutil.which", side_effect=fake_which):
            with patch(
                "kdenlive_face_mask.subprocess.run",
                return_value=type("R", (), {"returncode": 0, "stdout": expected.encode("utf-8")})(),
            ):
                result = read_from_clipboard()

        self.assertEqual(expected, result)

    def test_write_to_clipboard_uses_pbcopy_on_macos(self) -> None:
        def fake_which(cmd: str):
            return "/usr/bin/pbcopy" if cmd == "pbcopy" else None

        mock_process = unittest.mock.MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0

        with patch("kdenlive_face_mask.shutil.which", side_effect=fake_which):
            with patch("kdenlive_face_mask.subprocess.Popen", return_value=mock_process) as mock_popen:
                write_to_clipboard("<xml/>")

        called_cmd = mock_popen.call_args[0][0]
        self.assertEqual(["pbcopy"], called_cmd)

    def test_write_to_clipboard_prefers_powershell_on_windows(self) -> None:
        def fake_which(cmd: str):
            if cmd == "powershell.exe":
                return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            if cmd == "clip.exe":
                return r"C:\Windows\System32\clip.exe"
            return None

        mock_process = unittest.mock.MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0

        with patch("kdenlive_face_mask.shutil.which", side_effect=fake_which):
            with patch("kdenlive_face_mask.subprocess.Popen", return_value=mock_process) as mock_popen:
                write_to_clipboard("<xml/>")

        called_cmd = mock_popen.call_args[0][0]
        self.assertEqual("powershell.exe", called_cmd[0])
        self.assertIn("Set-Clipboard", called_cmd[-1])
        self.assertEqual("utf-8", mock_popen.call_args.kwargs["encoding"])

    def test_write_to_clipboard_falls_back_to_clip_exe_on_windows(self) -> None:
        def fake_which(cmd: str):
            return r"C:\Windows\System32\clip.exe" if cmd == "clip.exe" else None

        mock_process = unittest.mock.MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0

        with patch("kdenlive_face_mask.shutil.which", side_effect=fake_which):
            with patch("kdenlive_face_mask.subprocess.Popen", return_value=mock_process) as mock_popen:
                write_to_clipboard("<xml/>")

        called_cmd = mock_popen.call_args[0][0]
        self.assertEqual(["clip.exe"], called_cmd)

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

    def test_no_tracks_clear_generated_masks_in_replace_mode(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effects = root.findall("./clip[@id='15']/effects/effect")
        self.assertEqual([], effects)

    def test_no_tracks_preserve_existing_masks_in_keep_mode(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=False,
        )

        root = ET.fromstring(rewritten)
        effects = root.findall("./clip[@id='15']/effects/effect")
        self.assertEqual(1, len(effects))
        self.assertEqual(MASK_EFFECT_ID, effects[0].attrib.get("id"))

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

    def test_effect_count_matches_peak_concurrent_tracks(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [
                make_track(1, range(0, 2), 300.0),
                make_track(2, range(0, 11), 700.0),
                make_track(3, range(2, 4), 500.0),
            ],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effects = root.findall("./clip[@id='15']/effects/effect")
        self.assertEqual(2, len(effects))

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
            "0=0.0197917;1=0.0197917;2=0.0197917;3=0.0197917;4=0.0197917;5=0.0197917;6=0;7=0;8=0.0197917;9=0.0197917;10=0.0197917;11=0.0197917;12=0",
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

    def test_start_of_clip_track_begins_visible_on_frame_zero(self) -> None:
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
        self.assertEqual("0=0.0197917;1=0.0197917;2=0.0197917;3=0", size_x.text)

    def test_single_frame_track_at_clip_start_remains_visible(self) -> None:
        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [make_track(10, range(0, 1), 420.0)],
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
        self.assertEqual("0=0.0197917;1=0", size_x.text)

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
        self.assertEqual("0=0.0197917;4=0.0197917;7=0.0197917;8=0", size_x.text)

    def test_face_angle_emits_tilt_keyframes(self) -> None:
        track = FaceTrack(
            track_id=13,
            samples=[
                TrackSample(frame_index=0, cx=420.0, cy=140.0, half_w=38.0, half_h=54.0, angle_deg=0.0),
                TrackSample(frame_index=1, cx=422.0, cy=141.0, half_w=38.0, half_h=54.0, angle_deg=90.0),
                TrackSample(frame_index=2, cx=424.0, cy=142.0, half_w=38.0, half_h=54.0, angle_deg=-90.0),
            ],
        )

        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [track],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.5,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effect = root.find("./clip[@id='15']/effects/effect")
        self.assertIsNotNone(effect)
        tilt = effect.find("./property[@name='filter.Tilt']")
        self.assertIsNotNone(tilt)
        self.assertEqual("0=0.5;1=0.75;2=0.25;3=0.25", tilt.text)

    def test_face_angle_uses_requested_tilt_as_neutral_baseline(self) -> None:
        track = FaceTrack(
            track_id=14,
            samples=[
                TrackSample(frame_index=0, cx=420.0, cy=140.0, half_w=38.0, half_h=54.0, angle_deg=0.0),
                TrackSample(frame_index=1, cx=422.0, cy=141.0, half_w=38.0, half_h=54.0, angle_deg=90.0),
            ],
        )

        rewritten = rewrite_scene_with_tracks(
            SCENE_XML,
            [track],
            frame_width=1920,
            frame_height=1080,
            shape=0.38,
            tilt=0.25,
            replace_existing_masks=True,
        )

        root = ET.fromstring(rewritten)
        effect = root.find("./clip[@id='15']/effects/effect")
        self.assertIsNotNone(effect)
        tilt = effect.find("./property[@name='filter.Tilt']")
        self.assertIsNotNone(tilt)
        self.assertEqual("0=0.25;1=0.5;2=0.5", tilt.text)


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


class ValidationTests(unittest.TestCase):
    def test_build_mask_frames_rejects_non_positive_dimensions(self) -> None:
        track = FaceTrack(
            track_id=1,
            samples=[TrackSample(frame_index=0, cx=10.0, cy=10.0, half_w=2.0, half_h=2.0)],
        )

        with self.assertRaisesRegex(ValueError, "Invalid source video resolution"):
            build_mask_frames(track, total_frames=1, frame_width=0, frame_height=1080)

    def test_main_rejects_invalid_min_score_before_processing(self) -> None:
        exit_code = main(["--min-score", "1.5", "--log-level", "ERROR"])

        self.assertEqual(1, exit_code)

    def test_main_rejects_invalid_adaptive_keyframe_range(self) -> None:
        exit_code = main(
            [
                "--adaptive-keyframes",
                "--min-keyframe-fps",
                "60",
                "--max-keyframe-fps",
                "30",
                "--log-level",
                "ERROR",
            ]
        )

        self.assertEqual(1, exit_code)


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


class DoctorModeTests(unittest.TestCase):
    def test_main_doctor_reports_support_and_issue_or_pr_suggestion(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        def fake_build_detector(_det_size, _model_name, provider_mode):
            print(f"probe {provider_mode}", file=sys.stderr)
            if provider_mode == "cuda":
                return object(), ["CPUExecutionProvider"], "fake-cv2"
            if provider_mode == "openvino":
                return object(), ["OpenVINOExecutionProvider", "CPUExecutionProvider"], "fake-cv2"
            if provider_mode == "cpu":
                return object(), ["CPUExecutionProvider"], "fake-cv2"
            raise AssertionError(f"Unexpected provider mode {provider_mode}")

        with patch(
            "kdenlive_face_mask.installed_package_version",
            return_value="0.1.0",
        ):
            with patch(
                "kdenlive_face_mask.detect_host_environment",
                return_value={
                    "system": "Windows",
                    "machine": "AMD64",
                    "python_version": "3.12.9",
                    "platform": "Windows-11-10.0.22631-SP0",
                    "distribution": None,
                    "distribution_id": None,
                },
            ):
                with patch(
                    "kdenlive_face_mask.detect_onnxruntime_support",
                    return_value={
                        "import_ok": True,
                        "version": "1.25.1",
                        "available_providers": [
                            "CUDAExecutionProvider",
                            "OpenVINOExecutionProvider",
                            "CPUExecutionProvider",
                        ],
                        "error": None,
                    },
                ):
                    with patch(
                        "kdenlive_face_mask.detect_clipboard_support",
                        return_value={
                            "read_command": "powershell.exe",
                            "write_command": "clip.exe",
                            "available_read_commands": ["powershell.exe"],
                            "available_write_commands": ["clip.exe"],
                        },
                    ):
                        with patch(
                            "kdenlive_face_mask.build_detector",
                            side_effect=fake_build_detector,
                        ):
                            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                                exit_code = main(["--doctor", "--log-level", "ERROR"])

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())

        report = stdout.getvalue()
        self.assertIn("README bucket: Windows (x86_64)", report)
        self.assertIn("Preferred provider mode on this host: openvino", report)
        self.assertIn("Provider usability checks:", report)
        self.assertIn("- cuda: detected but unusable for detector init; fell back to CPUExecutionProvider", report)
        self.assertIn("- openvino: usable (OpenVINOExecutionProvider, CPUExecutionProvider)", report)
        self.assertIn("consider opening an issue or a PR", report)
        self.assertIn(
            "Suggested issue/PR title: validation report: Windows (x86_64) works on Windows (AMD64)",
            report,
        )
        self.assertIn(
            "ONNX Runtime providers: CUDAExecutionProvider, OpenVINOExecutionProvider, CPUExecutionProvider",
            report,
        )


class EndToEndCliSmokeTests(unittest.TestCase):
    def test_main_succeeds_when_no_tracks_are_generated(self) -> None:
        stdout = io.StringIO()

        with patch("kdenlive_face_mask.read_text_input", return_value=SCENE_XML):
            with patch(
                "kdenlive_face_mask.generate_tracks",
                return_value=([], 1920, 1080, ["CPUExecutionProvider"]),
            ):
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["--log-level", "ERROR"])

        self.assertEqual(0, exit_code)
        root = ET.fromstring(stdout.getvalue())
        effects = root.findall("./clip[@id='15']/effects/effect")
        self.assertEqual([], effects)

    def test_main_rewrites_xml_from_real_video_with_fake_detector(self) -> None:
        import cv2
        import numpy as np

        def create_smoke_video(directory: str, frame_count: int) -> str:
            candidates = [
                ("smoke.avi", "MJPG"),
                ("smoke.avi", "XVID"),
                ("smoke.mp4", "mp4v"),
            ]

            for file_name, fourcc_code in candidates:
                video_path = os.path.join(directory, file_name)
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*fourcc_code),
                    30.0,
                    (64, 48),
                )
                if not writer.isOpened():
                    continue

                for frame_index in range(frame_count):
                    frame = np.full((48, 64, 3), 30 + frame_index * 20, dtype=np.uint8)
                    writer.write(frame)
                writer.release()

                capture = cv2.VideoCapture(video_path)
                ok, _ = capture.read()
                capture.release()
                if ok:
                    return video_path

            self.skipTest("OpenCV could not create a readable smoke-test video on this platform")

        class FakeDetector:
            def __init__(self):
                self.calls = 0

            def get(self, _frame):
                offset = float(self.calls)
                self.calls += 1
                return [
                    type(
                        "FakeFace",
                        (),
                        {
                            "bbox": np.array([14.0 + offset, 10.0, 42.0 + offset, 38.0]),
                            "det_score": 0.99,
                            "kps": None,
                        },
                    )()
                ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = create_smoke_video(tmp_dir, frame_count=5)
            input_path = os.path.join(tmp_dir, "copied-clip.xml")
            output_path = os.path.join(tmp_dir, "rewritten-clip.xml")

            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write(
                    make_smoke_scene_xml(
                        video_path,
                        frame_count=5,
                        frame_width=64,
                        frame_height=48,
                        fps=30.0,
                    )
                )

            with patch(
                "kdenlive_face_mask.build_detector",
                return_value=(FakeDetector(), ["CPUExecutionProvider"], cv2),
            ):
                exit_code = main(
                    [
                        input_path,
                        "--output",
                        output_path,
                        "--provider-mode",
                        "cpu",
                        "--progress-every",
                        "0",
                        "--log-level",
                        "ERROR",
                    ]
                )

            with open(output_path, "r", encoding="utf-8") as handle:
                rewritten = handle.read()

        self.assertEqual(0, exit_code)

        root = ET.fromstring(rewritten)
        effects = root.findall("./clip[@id='26']/effects/effect")
        self.assertEqual(1, len(effects))
        self.assertEqual(MASK_EFFECT_ID, effects[0].attrib.get("id"))
        self.assertIsNotNone(effects[0].find("./property[@name='filter.Position X']"))


if __name__ == "__main__":
    unittest.main()