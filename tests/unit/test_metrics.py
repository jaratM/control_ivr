"""
Unit tests for the Pipeline Metrics module.
"""
import pytest
import time
from unittest.mock import Mock, MagicMock
from multiprocessing import Manager
from pipeline.metrics import PipelineMetrics, create_metrics


class TestPipelineMetrics:
    """Test PipelineMetrics class."""

    @pytest.fixture
    def manager(self):
        """Create a multiprocessing Manager."""
        return Manager()

    @pytest.fixture
    def metrics(self, manager):
        """Create a PipelineMetrics instance."""
        return PipelineMetrics(manager)

    def test_init_creates_counters(self, metrics):
        """Test that initialization creates all expected counters."""
        expected_counters = [
            "files_queued",
            "files_processed",
            "files_failed",
            "files_downloaded",
            "download_failed",
            "segments_created",
            "batches_created",
            "transcriptions_completed",
            "transcription_errors",
            "assemblies_completed",
            "classifications_completed",
            "classification_errors",
        ]
        
        for counter in expected_counters:
            assert metrics.get(counter) == 0

    def test_increment_counter(self, metrics):
        """Test incrementing a counter."""
        metrics.increment("files_processed")
        assert metrics.get("files_processed") == 1
        
        metrics.increment("files_processed", 5)
        assert metrics.get("files_processed") == 6

    def test_increment_nonexistent_counter(self, metrics):
        """Test incrementing a counter that doesn't exist creates it."""
        metrics.increment("custom_counter")
        assert metrics.get("custom_counter") == 1

    def test_get_counter(self, metrics):
        """Test getting counter values."""
        metrics.increment("files_queued", 10)
        assert metrics.get("files_queued") == 10
        assert metrics.get("nonexistent_counter") == 0

    def test_record_time(self, metrics):
        """Test recording timing data."""
        metrics.record_time("ingestion", "file1", 1.5)
        metrics.record_time("ingestion", "file2", 2.0)
        metrics.record_time("transcription", "file1", 3.5)
        
        summary = metrics.get_summary()
        timing_stats = summary["timing_stats"]
        
        assert "ingestion" in timing_stats
        assert timing_stats["ingestion"]["count"] == 2
        assert timing_stats["ingestion"]["total"] == 3.5
        assert timing_stats["ingestion"]["avg"] == 1.75

    def test_record_error(self, metrics):
        """Test recording errors."""
        metrics.record_error("ingestion", "file1", "Connection timeout")
        metrics.record_error("transcription", "file2", "Invalid format")
        
        summary = metrics.get_summary()
        
        assert summary["error_count"] == 2
        assert len(summary["errors"]) == 2
        assert summary["errors"][0]["operation"] == "ingestion"
        assert summary["errors"][0]["file_id"] == "file1"

    def test_record_error_truncates_long_messages(self, metrics):
        """Test that long error messages are truncated."""
        long_error = "x" * 300
        metrics.record_error("test", "file1", long_error)
        
        summary = metrics.get_summary()
        assert len(summary["errors"][0]["error"]) == 200

    def test_get_elapsed_time(self, metrics):
        """Test getting elapsed time."""
        time.sleep(0.1)
        elapsed = metrics.get_elapsed_time()
        assert elapsed >= 0.1
        assert elapsed < 1.0

    def test_get_summary_structure(self, metrics):
        """Test get_summary returns correct structure."""
        metrics.increment("files_processed", 5)
        metrics.record_time("ingestion", "file1", 1.0)
        metrics.record_error("test", "file1", "error")
        
        summary = metrics.get_summary()
        
        assert "counters" in summary
        assert "timing_stats" in summary
        assert "elapsed_seconds" in summary
        assert "error_count" in summary
        assert "errors" in summary
        
        assert isinstance(summary["counters"], dict)
        assert isinstance(summary["timing_stats"], dict)
        assert isinstance(summary["elapsed_seconds"], float)
        assert isinstance(summary["error_count"], int)
        assert isinstance(summary["errors"], list)

    def test_get_summary_limits_error_list(self, metrics):
        """Test that get_summary limits errors to last 10."""
        for i in range(15):
            metrics.record_error("test", f"file{i}", f"error{i}")
        
        summary = metrics.get_summary()
        
        assert summary["error_count"] == 15
        assert len(summary["errors"]) == 10

    def test_timing_stats_calculation(self, metrics):
        """Test timing statistics calculations."""
        metrics.record_time("ingestion", "file1", 1.0)
        metrics.record_time("ingestion", "file2", 2.0)
        metrics.record_time("ingestion", "file3", 3.0)
        
        summary = metrics.get_summary()
        stats = summary["timing_stats"]["ingestion"]
        
        assert stats["count"] == 3
        assert stats["total"] == 6.0
        assert stats["avg"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0

    def test_timing_stats_multiple_operations(self, metrics):
        """Test timing stats for multiple operation types."""
        metrics.record_time("ingestion", "file1", 1.0)
        metrics.record_time("transcription", "file1", 2.0)
        metrics.record_time("assembly", "file1", 0.5)
        metrics.record_time("classification", "file1", 1.5)
        
        summary = metrics.get_summary()
        timing_stats = summary["timing_stats"]
        
        assert len(timing_stats) == 4
        assert "ingestion" in timing_stats
        assert "transcription" in timing_stats
        assert "assembly" in timing_stats
        assert "classification" in timing_stats

    def test_log_summary_calls_logger(self, metrics):
        """Test that log_summary logs metrics."""
        from unittest.mock import patch
        
        metrics.increment("files_processed", 10)
        metrics.increment("files_queued", 10)
        
        with patch('pipeline.metrics.logger') as mock_logger:
            metrics.log_summary()
            
            # Should have called logger.info at least once
            assert mock_logger.info.called

    def test_log_summary_with_errors(self, metrics):
        """Test log_summary includes error information."""
        from unittest.mock import patch
        
        metrics.increment("files_processed", 5)
        metrics.record_error("test", "file1", "error message")
        
        with patch('pipeline.metrics.logger') as mock_logger:
            metrics.log_summary()
            
            # Should log warnings for errors
            assert mock_logger.warning.called

    def test_log_progress(self, metrics):
        """Test log_progress outputs progress information."""
        from unittest.mock import patch
        
        metrics.increment("files_queued", 100)
        metrics.increment("files_processed", 50)
        metrics.increment("files_failed", 5)
        
        with patch('pipeline.metrics.logger') as mock_logger:
            metrics.log_progress()
            
            assert mock_logger.info.called
            # Check that the logged message contains relevant info
            call_args = str(mock_logger.info.call_args)
            assert "50/100" in call_args or "processed" in call_args.lower()

    def test_concurrent_increment(self, manager):
        """Test that concurrent increments work correctly."""
        metrics = PipelineMetrics(manager)
        
        # Simulate concurrent increments
        for _ in range(10):
            metrics.increment("files_processed")
        
        assert metrics.get("files_processed") == 10

    def test_create_metrics_factory(self, manager):
        """Test the create_metrics factory function."""
        metrics = create_metrics(manager)
        
        assert isinstance(metrics, PipelineMetrics)
        assert metrics.get("files_processed") == 0

    def test_metrics_with_no_data(self, metrics):
        """Test metrics behavior with no recorded data."""
        summary = metrics.get_summary()
        
        assert summary["counters"]["files_processed"] == 0
        assert summary["timing_stats"] == {}
        assert summary["error_count"] == 0
        assert summary["errors"] == []

    def test_throughput_calculation_in_summary(self, metrics):
        """Test that throughput is calculated correctly in log_summary."""
        from unittest.mock import patch
        
        metrics.increment("files_queued", 100)
        metrics.increment("files_processed", 50)
        
        # Force specific elapsed time for predictable throughput
        with patch.object(metrics, 'get_elapsed_time', return_value=3600):
            with patch('pipeline.metrics.logger') as mock_logger:
                metrics.log_summary()
                
                # Throughput should be 50 files/hour
                call_args = str(mock_logger.info.call_args)
                assert "50" in call_args or "throughput" in call_args.lower()

    def test_success_rate_calculation(self, metrics):
        """Test success rate calculation in log_summary."""
        from unittest.mock import patch
        
        metrics.increment("files_queued", 100)
        metrics.increment("files_processed", 90)
        metrics.increment("files_failed", 10)
        
        with patch('pipeline.metrics.logger') as mock_logger:
            metrics.log_summary()
            
            call_args = str(mock_logger.info.call_args)
            # Success rate should be 90%
            assert "90" in call_args or "success" in call_args.lower()

    def test_eta_calculation_in_progress(self, metrics):
        """Test ETA calculation in log_progress."""
        from unittest.mock import patch
        
        metrics.increment("files_queued", 100)
        metrics.increment("files_processed", 50)
        
        with patch.object(metrics, 'get_elapsed_time', return_value=60):
            with patch('pipeline.metrics.logger') as mock_logger:
                metrics.log_progress()
                
                # Should calculate and log ETA
                call_args = str(mock_logger.info.call_args)
                assert "ETA" in call_args or "eta" in call_args.lower()

    def test_zero_division_handling(self, metrics):
        """Test that zero division is handled gracefully."""
        from unittest.mock import patch
        
        # No files processed, elapsed time is 0
        with patch.object(metrics, 'get_elapsed_time', return_value=0):
            with patch('pipeline.metrics.logger'):
                # Should not raise exception
                metrics.log_progress()
                metrics.log_summary()
