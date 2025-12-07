"""
Pipeline Metrics Collection Module

Simple, process-safe metrics tracking for the audio compliance pipeline.
Uses multiprocessing Manager for cross-process counter synchronization.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from multiprocessing import Manager
from loguru import logger
from datetime import datetime


class PipelineMetrics:
    """
    Thread/Process-safe metrics collector for pipeline monitoring.
    
    Usage:
        metrics = PipelineMetrics(manager)
        metrics.increment("files_processed")
        metrics.record_time("ingestion", 1.5)
        metrics.log_summary()
    """
    
    def __init__(self, manager: Manager):
        """Initialize metrics with a multiprocessing Manager."""
        self._counters = manager.dict()
        self._timings = manager.list()
        self._errors = manager.list()
        self._start_time = manager.Value('d', time.time())
        self._lock = manager.Lock()
        
        # Initialize counters
        counter_names = [
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
        for name in counter_names:
            self._counters[name] = 0
    
    def increment(self, counter_name: str, value: int = 1):
        """Increment a counter by the given value."""
        with self._lock:
            current = self._counters.get(counter_name, 0)
            self._counters[counter_name] = current + value
    
    def get(self, counter_name: str) -> int:
        """Get current value of a counter."""
        return self._counters.get(counter_name, 0)
    
    def record_time(self, operation: str, file_id: str, duration: float):
        """Record timing for an operation."""
        self._timings.append({
            "operation": operation,
            "file_id": file_id,
            "duration": duration,
            "timestamp": time.time()
        })
    
    def record_error(self, operation: str, file_id: str, error: str):
        """Record an error occurrence."""
        self._errors.append({
            "operation": operation,
            "file_id": file_id,
            "error": str(error)[:200],  # Truncate long errors
            "timestamp": time.time()
        })
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since metrics initialization."""
        return time.time() - self._start_time.value
    
    def get_summary(self) -> Dict:
        """Generate a summary of all metrics."""
        elapsed = self.get_elapsed_time()
        
        # Calculate timing statistics
        timing_stats = {}
        timings_list = list(self._timings)
        
        for op in ["ingestion", "transcription", "assembly", "classification"]:
            op_times = [t["duration"] for t in timings_list if t["operation"] == op]
            if op_times:
                timing_stats[op] = {
                    "count": len(op_times),
                    "total": sum(op_times),
                    "avg": sum(op_times) / len(op_times),
                    "min": min(op_times),
                    "max": max(op_times)
                }
        
        return {
            "counters": dict(self._counters),
            "timing_stats": timing_stats,
            "elapsed_seconds": elapsed,
            "error_count": len(self._errors),
            "errors": list(self._errors)[-10:]  # Last 10 errors
        }
    
    def log_summary(self):
        """Log a formatted summary of all metrics."""
        summary = self.get_summary()
        counters = summary["counters"]
        timing_stats = summary["timing_stats"]
        elapsed = summary["elapsed_seconds"]
        
        # Calculate throughput
        files_processed = counters.get("files_processed", 0)
        throughput = files_processed / elapsed * 3600 if elapsed > 0 else 0
        
        # Build summary message
        lines = [
            "",
            "=" * 70,
            "                     MANIFEST METRICS SUMMARY",
            "=" * 70,
            "",
            " FILE PROCESSING:",
            f"   • Files Queued:        {counters.get('files_queued', 0):>8}",
            f"   • Files Processed:     {counters.get('files_processed', 0):>8}",
            f"   • Files Failed:        {counters.get('files_failed', 0):>8}",
            f"   • Download Success:    {counters.get('files_downloaded', 0):>8}",
            f"   • Download Failed:     {counters.get('download_failed', 0):>8}",
            "",
            " PIPELINE STAGES:",
            f"   • Segments Created:    {counters.get('segments_created', 0):>8}",
            f"   • Batches Created:     {counters.get('batches_created', 0):>8}",
            f"   • Transcriptions:      {counters.get('transcriptions_completed', 0):>8}",
            f"   • Assemblies:          {counters.get('assemblies_completed', 0):>8}",
            f"   • Classifications:     {counters.get('classifications_completed', 0):>8}",
            "",
            " ERRORS:",
            f"   • Transcription Errors:    {counters.get('transcription_errors', 0):>5}",
            f"   • Classification Errors:   {counters.get('classification_errors', 0):>5}",
            f"   • Total Errors Logged:     {summary.get('error_count', 0):>5}",
            "",
        ]
        
        # Add timing statistics
        if timing_stats:
            lines.append("  TIMING STATISTICS (seconds):")
            for op, stats in timing_stats.items():
                lines.append(f"   {op.capitalize()}:")
                lines.append(f"      Count: {stats['count']:>6}  |  "
                           f"Total: {stats['total']:>8.2f}s  |  "
                           f"Avg: {stats['avg']:>6.3f}s  |  "
                           f"Range: [{stats['min']:.3f} - {stats['max']:.3f}]")
            lines.append("")
        
        # Add overall statistics
        lines.extend([
            " OVERALL:",
            f"   • Total Elapsed Time:  {elapsed:>10.2f} seconds ({elapsed/60:.1f} min)",
            f"   • Throughput:          {throughput:>10.1f} files/hour",
            f"   • Success Rate:        {(files_processed / max(counters.get('files_queued', 1), 1) * 100):>10.1f}%",
            "",
            "=" * 70,
            ""
        ])
        
        # Log as single message
        logger.info("\n".join(lines))
        
        # Log recent errors if any
        if summary["errors"]:
            logger.warning(f"Recent errors ({len(summary['errors'])} shown):")
            for err in summary["errors"]:
                logger.warning(f"  [{err['operation']}] {err['file_id']}: {err['error']}")
    
    def log_progress(self, interval_seconds: float = 60.0):
        """Log a brief progress update (call periodically from main process)."""
        counters = dict(self._counters)
        elapsed = self.get_elapsed_time()
        
        processed = counters.get("files_processed", 0)
        failed = counters.get("files_failed", 0)
        queued = counters.get("files_queued", 0)
        
        remaining = queued - processed - failed
        rate = processed / elapsed if elapsed > 0 else 0
        eta = remaining / rate if rate > 0 else 0
        
        logger.info(
            f" Progress: {processed}/{queued} processed "
            f"({failed} failed) | "
            f"Rate: {rate:.1f}/s | "
            f"ETA: {eta/60:.1f} min"
        )


# Convenience function to create metrics with an existing manager
def create_metrics(manager: Manager) -> PipelineMetrics:
    """Factory function to create PipelineMetrics instance."""
    return PipelineMetrics(manager)

