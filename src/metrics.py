import threading
from typing import List

class StreamMetrics:
    """Thread-safe metrics collection for a single stream."""
    def __init__(self):
        self._lock = threading.Lock()
        self._ages: List[float] = []
        self._preprocessed = 0

    def record_age(self, age_ms: float) -> None:
        with self._lock:
            self._ages.append(age_ms)

    def record_preprocess(self) -> None:
        with self._lock:
            self._preprocessed += 1

    def pop_stats(self) -> dict:
        """Returns and resets the metrics."""
        with self._lock:
            ages = self._ages
            self._ages = []
            preprocessed = self._preprocessed
            self._preprocessed = 0

        if not ages:
            return {"age_ms_min": 0, "age_ms_max": 0, "age_ms_avg": 0, "age_ms_p99": 0, "preprocessed": 0}
        
        ages.sort()
        idx_p99 = min(len(ages) - 1, int(0.99 * len(ages)))
        return {
            "age_ms_min": round(ages[0], 2),
            "age_ms_max": round(ages[-1], 2),
            "age_ms_avg": round(sum(ages) / len(ages), 2),
            "age_ms_p99": round(ages[idx_p99], 2),
            "preprocessed": preprocessed
        }
