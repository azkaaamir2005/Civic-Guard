import cv2
import numpy as np
import time
from typing import Dict, Tuple, Any

class VisionEngine:
    def __init__(self, youngs_modulus_gpa: float = 30.0):
        self.youngs_modulus = youngs_modulus_gpa * 1e9
        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid = (8, 8)
        self.canny_low = 50
        self.canny_high = 150
        self.prev_gray_small = None
        self.cached_norm_stress = None
        self.frame_counter = 0
        
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_grid
        )
        
        self.build_info = cv2.getBuildInformation()
        self.is_cool_accelerated = any(
            token in self.build_info for token in ["NEON", "KleidiCV", "HAL", "AArch64"]
        )

    def check_hardware_acceleration(self) -> Dict[str, Any]:
        return {
            "cool_accelerated": self.is_cool_accelerated,
            "opencv_version": cv2.__version__,
            "cpu_optimization_flags": [
                line for line in self.build_info.split("\n") if "CPU" in line or "Use" in line
            ][:5]
        }

    def update_parameters(self, clahe_clip: float, canny_low: int, canny_high: int) -> None:
        self.clahe_clip_limit = max(0.5, min(10.0, float(clahe_clip)))
        self.canny_low = max(10, min(255, int(canny_low)))
        self.canny_high = max(self.canny_low + 10, min(255, int(canny_high)))
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_grid
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if frame is None or frame.size == 0:
            return np.array([]), np.array([]), {
                "latency_ms": 0.0, "max_stress_mpa": 0.0, "crack_density_pct": 0.0, 
                "hazard_index": 0.0, "blur_score": 0.0, "contrast_score": 0.0, 
                "structure_status": "NOMINAL"
            }

        start_time = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]
        self.frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)

        # Downscale grid to 200px width for max throughput
        target_w = 200
        scale = target_w / max(orig_w, 1)
        if scale < 1.0:
            target_h = int(orig_h * scale)
            small_enhanced = cv2.resize(enhanced_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            small_enhanced = enhanced_gray
            target_h, target_w = orig_h, orig_w

        edges_small = cv2.Canny(small_enhanced, self.canny_low, self.canny_high)
        crack_density = float(np.sum(edges_small > 0) / edges_small.size)
        edges = cv2.resize(edges_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else edges_small

        max_stress_mpa = 0.0
        has_motion = False

        # Calculate Optical Flow every 3rd frame to optimize latency
        if self.frame_counter % 3 == 0 and self.prev_gray_small is not None and self.prev_gray_small.shape == small_enhanced.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray_small, small_enhanced, None, 
                0.5, 1, 7, 1, 5, 1.1, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(magnitude))

            if mean_mag > 0.05:
                has_motion = True
                strain = magnitude / (target_w * 0.01)
                stress_pa = self.youngs_modulus * strain
                max_stress_mpa = float(np.max(stress_pa) / 1e6)
                self.cached_norm_stress = np.clip((stress_pa / 1e6) / 100.0 * 255, 0, 255).astype(np.uint8)

        elif self.cached_norm_stress is not None and self.cached_norm_stress.shape == small_enhanced.shape:
            has_motion = True
            norm_stress = self.cached_norm_stress

        if not has_motion:
            gx = cv2.Sobel(small_enhanced, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(small_enhanced, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(gx, gy)
            
            static_stress_map = (grad_mag / 1024.0) * (1.0 + crack_density * 10.0) * 45.0
            max_stress_mpa = float(np.max(static_stress_map))
            norm_stress = np.clip(static_stress_map / 100.0 * 255, 0, 255).astype(np.uint8)
        else:
            norm_stress = self.cached_norm_stress if self.cached_norm_stress is not None else np.zeros_like(small_enhanced)

        self.prev_gray_small = small_enhanced.copy()

        heatmap_small = cv2.applyColorMap(norm_stress, cv2.COLORMAP_JET)
        heatmap_raw = cv2.resize(heatmap_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else heatmap_small

        base_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        stress_heatmap = cv2.addWeighted(base_bgr, 0.6, heatmap_raw, 0.4, 0)

        blur_score = float(cv2.Laplacian(small_enhanced, cv2.CV_64F).var())
        contrast_score = float(small_enhanced.std())

        norm_stress_val = min(1.0, max_stress_mpa / 100.0)
        shi = min(1.0, max(0.0, float((0.6 * crack_density * 5.0) + (0.4 * norm_stress_val))))

        if shi > 0.65 or max_stress_mpa > 70.0:
            structure_status = "CRITICAL"
        elif shi > 0.30 or max_stress_mpa > 35.0:
            structure_status = "ELEVATED RISK"
        else:
            structure_status = "NOMINAL"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metrics = {
            "latency_ms": round(latency_ms, 2),
            "max_stress_mpa": round(max_stress_mpa, 2),
            "crack_density_pct": round(crack_density * 100.0, 2),
            "hazard_index": round(shi, 2),
            "blur_score": round(blur_score, 2),
            "contrast_score": round(contrast_score, 2),
            "structure_status": structure_status
        }

        return edges, stress_heatmap, metrics