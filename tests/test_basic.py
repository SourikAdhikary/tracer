"""Basic tests for Tracer modules."""

import numpy as np
import pytest

from tracer.config import Config
from tracer.schemas import Detection, FrameResult, AuditReport
from tracer.video import frame_timestamp
from tracer.crop import crop_detection
from tracer.qoe import QoEScorer


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.pipeline.extraction_fps == 1.0
        assert config.pipeline.frame_size == 1000
        assert config.models.scout_token_budget == 70
        assert config.models.auditor_token_budget == 560

    def test_ensure_dirs(self, tmp_path):
        config = Config()
        config.paths.output_dir = tmp_path / "output"
        config.paths.crops_dir = tmp_path / "output" / "crops"
        config.ensure_dirs()
        assert config.paths.output_dir.exists()
        assert config.paths.crops_dir.exists()


class TestSchemas:
    def test_detection(self):
        det = Detection(
            brand="Emirates",
            box_2d=[100, 200, 300, 400],
            confidence=0.95,
            label="Emirates_Chest",
        )
        assert det.brand == "Emirates"
        assert det.box_2d == [100, 200, 300, 400]

    def test_frame_result(self):
        result = FrameResult(
            frame_index=42,
            timestamp="00:00:42",
            detections=[
                Detection(brand="Emirates", box_2d=[100, 200, 300, 400], confidence=0.9),
            ],
        )
        assert len(result.detections) == 1

    def test_audit_report(self):
        report = AuditReport(
            match_id="test",
            brands_tracked=["Emirates"],
            frames_extracted=100,
        )
        assert report.frames_extracted == 100


class TestVideo:
    def test_frame_timestamp(self):
        assert frame_timestamp(0) == "00:00:00"
        assert frame_timestamp(60) == "00:01:00"
        assert frame_timestamp(3661) == "01:01:01"


class TestCrop:
    def test_crop_detection(self, tmp_path):
        # Create a dummy 100x100 frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        output_path = tmp_path / "test_crop.png"

        result = crop_detection(frame, [10, 10, 50, 50], output_path, padding=0)
        assert output_path.exists()


class TestQoE:
    def test_clarity(self):
        config = Config()
        scorer = QoEScorer(config)

        # Sharp image (random noise has high Laplacian variance)
        sharp = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        clarity = scorer._compute_clarity(sharp)
        assert clarity > 0.0

        # Blurred image
        blurred = np.zeros((50, 50, 3), dtype=np.uint8)
        blurred[:] = 128
        clarity_blurred = scorer._compute_clarity(blurred)
        assert clarity_blurred < clarity

    def test_size(self):
        config = Config()
        scorer = QoEScorer(config)

        # Large box
        size_large = scorer._compute_size([0, 0, 500, 500])  # 25% of frame
        # Small box
        size_small = scorer._compute_size([0, 0, 10, 10])  # 0.01% of frame
        assert size_large > size_small

    def test_context(self):
        config = Config()
        scorer = QoEScorer(config)

        assert scorer._assess_context("Goal celebration") == 1.0
        assert scorer._assess_context("Active play, midfield") == 0.5
        assert scorer._assess_context("") == 0.5
