"""Tests for pipeline orchestrator."""
import pytest
from unittest.mock import patch, MagicMock

from app.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator:
    def test_orchestrator_initializes(self):
        """Test orchestrator can be created."""
        orch = PipelineOrchestrator()
        assert orch is not None

    def test_orchestrator_tracks_current_stage(self):
        """Test stage tracking is initialized."""
        orch = PipelineOrchestrator()
        assert hasattr(orch, '_current_stage')

    def test_orchestrator_has_stages(self):
        """Test STAGES list is defined and non-empty."""
        assert hasattr(PipelineOrchestrator, 'STAGES')
        assert len(PipelineOrchestrator.STAGES) > 0

    def test_orchestrator_stages_include_all_phases(self):
        """Test all 7 pipeline stages are listed."""
        expected = {"etl", "stats", "supply_chain", "ml", "capacity", "sensing", "sop"}
        assert set(PipelineOrchestrator.STAGES) == expected

    def test_orchestrator_accepts_progress_callback(self):
        """Test custom progress callback is stored."""
        cb = MagicMock()
        orch = PipelineOrchestrator(on_progress=cb)
        assert orch.on_progress is cb

    def test_orchestrator_default_callback_is_noop(self):
        """Test default progress callback does not raise."""
        orch = PipelineOrchestrator()
        # Should not raise
        orch.on_progress("etl", "running", {})
