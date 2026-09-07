"""Tests for single-channel non-depth (mono) video features.

Mono covers greyscale camera images such as the infrared imagers of a stereo
depth camera. They share depth's ``(H, W, 1)`` shape but are intensities, not
distances, so they must bypass depth quantization and be stored bit-exact.

Covers:
- ``hw_to_dataset_features`` opt-in mono flagging (and that it is opt-in).
- ``LeRobotDatasetMetadata.mono_keys`` detection.
- Encoder-bucket routing through the dataset writer and streaming encoder.
- Bit-exactness of the mono encoder defaults.
"""

from pathlib import Path

import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import av
import numpy as np

from lerobot.configs import DepthEncoderConfig, MonoEncoderConfig, RGBEncoderConfig
from lerobot.configs.video import encoder_config_from_video_info, mono_encoder_defaults
from lerobot.utils.feature_utils import hw_to_dataset_features
from tests.fixtures.constants import (
    DEFAULT_FPS,
    DUMMY_CAMERA_FEATURES,
    DUMMY_CAMERA_FEATURES_WITH_DEPTH_AND_MONO,
    DUMMY_DEPTH_CAMERA_FEATURES,
    DUMMY_MONO_CAMERA_FEATURES,
    DUMMY_REPO_ID,
)
from tests.fixtures.dataset_factories import add_frames

# Fixture feature dicts use bare camera keys, not the observation.images.* prefix.
RGB_KEY = next(iter(DUMMY_CAMERA_FEATURES))
DEPTH_KEY = next(iter(DUMMY_DEPTH_CAMERA_FEATURES))
MONO_KEY = next(iter(DUMMY_MONO_CAMERA_FEATURES))


# ── 1. Feature flagging ──────────────────────────────────────────────


class TestHwToDatasetFeaturesMono:
    """``mono_keys`` opts a single-channel camera out of the depth flag."""

    def test_mono_key_is_flagged_mono_not_depth(self):
        features = hw_to_dataset_features(
            {"ir_left": (480, 640, 1)}, prefix="observation", mono_keys=["ir_left"]
        )
        info = features["observation.images.ir_left"]["info"]
        assert info["is_mono"] is True
        assert info["is_depth_map"] is False

    def test_omitting_mono_keys_is_unchanged(self):
        """Default behaviour must stay byte-identical for existing callers."""
        without = hw_to_dataset_features({"cam": (480, 640, 1)}, prefix="observation")
        explicit_none = hw_to_dataset_features({"cam": (480, 640, 1)}, prefix="observation", mono_keys=None)
        assert without == explicit_none
        assert without["observation.images.cam"]["info"] == {"is_depth_map": True}

    def test_rgb_unaffected_by_mono_keys(self):
        features = hw_to_dataset_features(
            {"rgb": (480, 640, 3), "ir": (480, 640, 1)},
            prefix="observation",
            mono_keys=["ir"],
        )
        assert features["observation.images.rgb"]["info"] == {"is_depth_map": False}

    def test_unknown_mono_key_raises(self):
        with pytest.raises(ValueError, match="are not camera features"):
            hw_to_dataset_features({"cam": (480, 640, 1)}, prefix="observation", mono_keys=["nope"])

    def test_three_channel_mono_key_raises(self):
        with pytest.raises(ValueError, match="must be single-channel"):
            hw_to_dataset_features({"cam": (480, 640, 3)}, prefix="observation", mono_keys=["cam"])


# ── 2. Encoder dispatch from persisted info ──────────────────────────


class TestEncoderConfigFromVideoInfo:
    @pytest.mark.parametrize(
        "info,expected",
        [
            ({"is_depth_map": True}, DepthEncoderConfig),
            ({"is_mono": True}, MonoEncoderConfig),
            ({"is_depth_map": False, "is_mono": True}, MonoEncoderConfig),
            ({"is_depth_map": False}, RGBEncoderConfig),
            ({}, RGBEncoderConfig),
        ],
    )
    def test_dispatch(self, info, expected):
        assert isinstance(encoder_config_from_video_info(info), expected)

    def test_depth_wins_over_mono(self):
        """A feature marked both ways is depth; mono must not hijack it."""
        cfg = encoder_config_from_video_info({"is_depth_map": True, "is_mono": True})
        assert isinstance(cfg, DepthEncoderConfig)


# ── 3. Metadata detection ────────────────────────────────────────────


class TestMonoKeysMetadata:
    def test_mono_keys_separates_from_depth(self, tmp_path, features_factory):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = features_factory(camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH_AND_MONO)
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=tmp_path / "ds",
            use_videos=True,
        )
        assert MONO_KEY in dataset.meta.mono_keys
        assert MONO_KEY not in dataset.meta.depth_keys
        assert DEPTH_KEY in dataset.meta.depth_keys
        assert DEPTH_KEY not in dataset.meta.mono_keys
        assert RGB_KEY not in dataset.meta.mono_keys


# ── 4. Writer routing ────────────────────────────────────────────────


class TestMonoEncoderRouting:
    NUM_FRAMES = 5

    def test_streaming_routes_mono_to_mono_encoder(self, tmp_path, features_factory):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = features_factory(camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH_AND_MONO)
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=tmp_path / "ds",
            use_videos=True,
            streaming_encoding=True,
        )
        add_frames(dataset, num_frames=self.NUM_FRAMES)

        threads = dataset.writer._streaming_encoder._threads
        assert isinstance(threads[MONO_KEY].video_encoder, MonoEncoderConfig)
        assert isinstance(threads[DEPTH_KEY].video_encoder, DepthEncoderConfig)
        assert not isinstance(threads[RGB_KEY].video_encoder, (MonoEncoderConfig, DepthEncoderConfig))

        dataset.save_episode()
        dataset.finalize()

    def test_encoder_for_picks_each_bucket(self, tmp_path, features_factory):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = features_factory(camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH_AND_MONO)
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=tmp_path / "ds",
            use_videos=True,
        )
        writer = dataset.writer
        assert writer._encoder_for(MONO_KEY) is writer._mono_encoder
        assert writer._encoder_for(DEPTH_KEY) is writer._depth_encoder
        assert writer._encoder_for(RGB_KEY) is writer._rgb_encoder


# ── 5. Losslessness ──────────────────────────────────────────────────


class TestMonoLossless:
    """The whole point of the mono bucket: pixels survive the round-trip."""

    def test_end_to_end_through_writer_is_bit_exact(self, tmp_path, features_factory):
        """Frames written through the dataset decode back byte-for-byte.

        This is the guarantee offline stereo matching depends on, and it exercises
        the real streaming-encoder path rather than a standalone encode.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        num_frames = 10
        mono_features = {
            MONO_KEY: {
                "shape": (64, 96, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": False, "is_mono": True},
            }
        }
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features_factory(motor_features={}, camera_features=mono_features),
            root=tmp_path / "ds",
            use_videos=True,
            streaming_encoding=True,
        )

        rng = np.random.default_rng(0)
        sent = [rng.integers(0, 256, (64, 96, 1), dtype=np.uint8) for _ in range(num_frames)]
        for frame in sent:
            dataset.add_frame({"task": "test", MONO_KEY: frame})
        dataset.save_episode()
        dataset.finalize()

        video_path = dataset.root / dataset.meta.get_video_file_path(0, MONO_KEY)
        assert video_path.exists()
        with av.open(str(video_path)) as container:
            decoded = [f.to_ndarray(format="gray") for f in container.decode(video=0)]

        assert len(decoded) == num_frames
        for original, roundtripped in zip(sent, decoded, strict=True):
            assert np.array_equal(original[..., 0], roundtripped)

    def test_defaults_are_bit_exact(self, tmp_path):
        cfg = mono_encoder_defaults()
        assert cfg.pix_fmt == "gray"

        h, w, n = 64, 96, 12
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (h, w), dtype=np.uint8) for _ in range(n)]

        path = Path(tmp_path) / "mono.mkv"
        opts = cfg.get_codec_options(as_strings=True)
        with av.open(str(path), "w") as out:
            stream = out.add_stream(cfg.vcodec, rate=30, options=opts)
            stream.width, stream.height, stream.pix_fmt = w, h, cfg.pix_fmt
            for f in frames:
                out.mux(stream.encode(av.VideoFrame.from_ndarray(f, format="gray")))
            out.mux(stream.encode(None))

        with av.open(str(path)) as inp:
            decoded = [f.to_ndarray(format="gray") for f in inp.decode(video=0)]

        assert len(decoded) == n
        for original, roundtripped in zip(frames, decoded, strict=True):
            assert np.array_equal(original, roundtripped)
