import argparse
import time
import json
import logging
import threading
import signal
import sys

from src.slot import LatestSlot
from src.capture import CaptureThread
from src.quality import QualityGate
from src.metrics import StreamMetrics
from src.pipeline import ConsumerThread

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="Real-Time Video Pipeline")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--numpy", action="store_true", help="Use hand-written numpy preprocessing instead of cv2.dnn")
    args = parser.parse_args()

    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        logger.info("SIGINT received, shutting down...")
        stop_event.set()
        
    signal.signal(signal.SIGINT, signal_handler)

    slots = {
        "cam_a": LatestSlot(),
        "cam_b": LatestSlot()
    }
    
    metrics = {
        "cam_a": StreamMetrics(),
        "cam_b": StreamMetrics()
    }
    
    qg = QualityGate()
    
    caps = [
        CaptureThread(args.input, slots["cam_a"], "cam_a", stop_event),
        CaptureThread(args.input, slots["cam_b"], "cam_b", stop_event)
    ]
    
    consumer = ConsumerThread(slots, qg, metrics, stop_event, use_numpy=args.numpy)
    
    for c in caps:
        c.start()
    consumer.start()
    
    logger.info("Pipeline started. Press Ctrl+C to stop.")
    
    try:
        while not stop_event.is_set():
            # Wait for 5 seconds or until stop_event is set
            stopped = stop_event.wait(timeout=5.0)
            if stopped:
                break
                
            stats = {}
            for sid in slots:
                cap_stats = {}
                for c in caps:
                    if c.source_id == sid:
                        cap_stats = c.get_stats()
                        break
                        
                qg_stats = qg.get_counters(sid)
                sm_stats = metrics[sid].pop_stats()
                
                with slots[sid]._lock:
                    dropped = slots[sid].frames_dropped
                
                stats[sid] = {
                    **cap_stats,
                    "frames_dropped": dropped,
                    **qg_stats,
                    **sm_stats
                }
            
            print(json.dumps(stats, indent=2))
            
    except KeyboardInterrupt:
        pass
        
    for c in caps:
        c.join(timeout=2.0)
    consumer.join(timeout=2.0)
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
