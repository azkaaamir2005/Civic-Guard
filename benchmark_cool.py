import time
import cv2
import numpy as np
import json
import psutil

def run_benchmark():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    latencies = []
    cpu_usages = []

    # Warmup
    for _ in range(20):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _ = clahe.apply(gray)

    # Benchmark Execution (1000 iterations)
    for _ in range(1000):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
        _ = cv2.Canny(enhanced, 50, 150)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)
        cpu_usages.append(psutil.cpu_percent())

    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency
    avg_cpu = float(np.mean(cpu_usages))

    results = {
        "architecture": "AWS Graviton3 (c7g.xlarge - Arm64)",
        "opencv_version": cv2.__version__,
        "average_latency_ms": round(avg_latency, 3),
        "throughput_fps": round(fps, 2),
        "average_cpu_percent": round(avg_cpu, 2),
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("=== BENCHMARK COMPLETE ===")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    run_benchmark()