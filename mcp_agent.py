import os
import json
import time
import boto3
import google.genai as genai
from typing import Dict, Any, List

class AgenticVisionController:
    def __init__(self, sns_topic_arn: str = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.sns_topic_arn = sns_topic_arn
        self.sns_client = boto3.client("sns", region_name="us-east-1") if sns_topic_arn else None
        self.decision_history: List[Dict[str, Any]] = []

        if self.gemini_key:
            self.genai_client = genai.Client(api_key=self.gemini_key)
        else:
            self.genai_client = None

    def evaluate_and_act(
        self, metrics: Dict[str, float], vision_engine, is_autonomous: bool = True
    ) -> Dict[str, Any]:
        """Closed-loop decision cycle: Evaluates metrics, invokes tools, dispatches alerts."""
        if not is_autonomous:
            return {
                "status": "OVERRIDE",
                "timestamp": time.strftime("%H:%M:%S"),
                "reason": "Human-in-the-Loop Override Active. Agent paused."
            }

        start_time = time.time()
        action_record = {
            "timestamp": time.strftime("%H:%M:%S"),
            "status": "ACTIVE",
            "input_metrics": metrics,
            "tool_calls": [],
            "fallback_triggered": False,
            "action_summary": "",
            "rationale": ""
        }

        # Dynamic trigger conditions calibrated to incoming telemetry ranges
        needs_clahe_boost = metrics.get("contrast_score", 50.0) < 45.0
        needs_edge_tightening = metrics.get("blur_score", 150.0) < 120.0
        critical_stress = metrics.get("max_stress_mpa", 0.0) > 35.0 or metrics.get("structure_status") == "CRITICAL"

        # If conditions are completely nominal, return structured active monitor state
        if not (needs_clahe_boost or needs_edge_tightening or critical_stress):
            action_record["status"] = "NOMINAL_MONITORING"
            action_record["action_summary"] = "No parameter adjustments needed."
            action_record["rationale"] = "Telemetry within optimal structural operating bounds."
            self.decision_history.append(action_record)
            return action_record

        # LLM Reasoning & Tool Call Execution with Heuristic Fallback Guard
        try:
            if self.genai_client:
                prompt = f"""
                You are CivicGuard Autonomous Vision Agent.
                Analyze real-time structural inspection telemetry:
                - Contrast Score: {metrics.get('contrast_score', 0):.2f} (Target > 45)
                - Blur Score: {metrics.get('blur_score', 0):.2f} (Target > 120)
                - Max Structural Stress: {metrics.get('max_stress_mpa', 0):.2f} MPa (Threshold 35 MPa)
                - Structural Health Index: {metrics.get('hazard_index', 0):.2f}

                Formulate an operational decision. Return JSON ONLY with keys:
                "clahe_clip": float (0.5 to 8.0),
                "canny_low": int (10 to 150),
                "canny_high": int (50 to 255),
                "trigger_alert": bool,
                "rationale": str
                """
                
                response = self.genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                
                decision = json.loads(response.text)
                new_clahe = float(decision.get("clahe_clip", 3.5))
                new_low = int(decision.get("canny_low", 30))
                new_high = int(decision.get("canny_high", 120))
                
                # Action 1: Reconfigure OpenCV Vision Engine Parameters
                vision_engine.update_parameters(new_clahe, new_low, new_high)
                action_record["tool_calls"].append({
                    "tool": "update_opencv_parameters",
                    "status": "SUCCESS",
                    "params": {"clahe_clip": new_clahe, "canny_low": new_low, "canny_high": new_high}
                })
                
                # Action 2: Trigger AWS SNS Emergency Dispatch if threshold exceeded
                if decision.get("trigger_alert") or critical_stress:
                    alert_sent = self._dispatch_sns_alert(metrics.get("max_stress_mpa", 0.0))
                    action_record["tool_calls"].append({
                        "tool": "dispatch_aws_sns_alert",
                        "status": "DISPATCHED" if alert_sent else "SIMULATED_ALERT",
                        "params": {"stress_mpa": metrics.get("max_stress_mpa", 0.0)}
                    })

                action_record["status"] = "ACTION_EXECUTED"
                action_record["action_summary"] = f"Adjusted CLAHE={new_clahe:.1f}, Canny=({new_low}, {new_high})"
                action_record["rationale"] = decision.get("rationale", "Optimized via Gemini Vision Agent.")

            else:
                raise ValueError("Gemini Client not initialized. Running Heuristic Engine.")

        except Exception as e:
            # Fallback Execution Loop for System Resilience
            action_record["fallback_triggered"] = True
            new_clahe = 4.0 if needs_clahe_boost else 2.0
            new_low = 20 if needs_edge_tightening else 50
            new_high = 140
            
            vision_engine.update_parameters(new_clahe, new_low, new_high)
            action_record["tool_calls"].append({
                "tool": "update_opencv_parameters",
                "status": "FALLBACK_SUCCESS",
                "params": {"clahe_clip": new_clahe, "canny_low": new_low, "canny_high": new_high}
            })

            if critical_stress:
                alert_sent = self._dispatch_sns_alert(metrics.get("max_stress_mpa", 0.0))
                action_record["tool_calls"].append({
                    "tool": "dispatch_aws_sns_alert",
                    "status": "FALLBACK_DISPATCHED" if alert_sent else "SIMULATED_ALERT",
                    "params": {"stress_mpa": metrics.get("max_stress_mpa", 0.0)}
                })

            action_record["status"] = "FALLBACK_EXECUTED"
            action_record["action_summary"] = f"Fallback applied: CLAHE={new_clahe}, Canny=({new_low}, {new_high})"
            action_record["rationale"] = f"Resilience Heuristic Activated: {str(e)[:60]}"

        self.decision_history.append(action_record)
        return action_record

    def _dispatch_sns_alert(self, stress_mpa: float) -> bool:
        """Sends emergency notification to AWS SNS."""
        if self.sns_client and self.sns_topic_arn:
            try:
                self.sns_client.publish(
                    TopicArn=self.sns_topic_arn,
                    Subject="[EMERGENCY] CivicGuard Structural Stress Limit Exceeded",
                    Message=f"CRITICAL WARNING: Measured yield stress reached {stress_mpa:.2f} MPa."
                )
                return True
            except Exception:
                return False
        return False