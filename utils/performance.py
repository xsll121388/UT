"""Performance monitoring utilities for UT."""
from __future__ import annotations
import time
from typing import Dict, List
from dataclasses import dataclass, field
import statistics


@dataclass
class TimerStats:
    """Statistics for a named timer."""
    name: str
    count: int = 0
    total: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    
    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
    
    @property
    def avg_ms(self) -> float:
        return self.avg * 1000
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'count': self.count,
            'avg_ms': round(self.avg_ms, 2),
            'min_ms': round(self.min * 1000, 2) if self.min != float('inf') else 0,
            'max_ms': round(self.max * 1000, 2),
            'total_ms': round(self.total * 1000, 2)
        }


class PerformanceMonitor:
    """
    Performance monitor for tracking operation timings.
    
    Usage:
        monitor = PerformanceMonitor()
        
        # In code
        with monitor.timer("mel_extraction"):
            mel = extract_mel(audio)
        
        # Later
        stats = monitor.get_stats("mel_extraction")
        print(f"Average: {stats.avg_ms:.2f}ms")
        
        # Generate report
        monitor.report()
    """
    
    def __init__(self):
        self._timers: Dict[str, TimerStats] = {}
        self._active_timers: Dict[str, float] = {}
    
    def timer(self, name: str) -> TimerContext:
        """Create a timer context manager."""
        return TimerContext(self, name)
    
    def start(self, name: str):
        """Start a named timer."""
        if name in self._active_timers:
            raise RuntimeError(f"Timer '{name}' is already running")
        self._active_timers[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop a named timer and record the elapsed time."""
        if name not in self._active_timers:
            raise RuntimeError(f"Timer '{name}' was not started")
        
        start = self._active_timers.pop(name)
        elapsed = time.perf_counter() - start
        
        if name not in self._timers:
            self._timers[name] = TimerStats(name=name)
        
        stats = self._timers[name]
        stats.count += 1
        stats.total += elapsed
        stats.min = min(stats.min, elapsed)
        stats.max = max(stats.max, elapsed)
        
        return elapsed
    
    def get_stats(self, name: str) -> TimerStats | None:
        """Get statistics for a named timer."""
        return self._timers.get(name)
    
    def get_all_stats(self) -> Dict[str, TimerStats]:
        """Get all timer statistics."""
        return self._timers.copy()
    
    def report(self, min_count: int = 1) -> str:
        """Generate a performance report."""
        if not self._timers:
            return "No performance data collected"
        
        lines = [
            "=" * 70,
            "PERFORMANCE REPORT",
            "=" * 70,
            f"{'Operation':<40} {'Count':>6} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}",
            "-" * 70
        ]
        
        # Sort by average time (descending)
        sorted_timers = sorted(
            self._timers.values(),
            key=lambda s: s.avg,
            reverse=True
        )
        
        for stats in sorted_timers:
            if stats.count >= min_count:
                lines.append(
                    f"{stats.name:<40} {stats.count:>6} {stats.avg_ms:>10.2f} "
                    f"{stats.min*1000:>10.2f} {stats.max*1000:>10.2f}"
                )
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def reset(self):
        """Reset all timers."""
        self._timers.clear()
        self._active_timers.clear()


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
    
    def __enter__(self):
        self.monitor.start(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop(self.name)
        return False


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


def time_operation(name: str):
    """Decorator for timing function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            with monitor.timer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
