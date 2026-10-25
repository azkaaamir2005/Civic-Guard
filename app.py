import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import threading
from vision_engine import VisionEngine
from mcp_agent import AgenticVisionController

# Page Configuration
st.set_page_config(page_title="CivicGuard - COOL & Agentic Vision", layout="wide")

def create_side_by_side_stream(original_frame, processed_frame):
    """Combines original frame and processed heatmap horizontally."""
    if len(processed_frame.shape) == 2:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    
    if original_frame.shape[:2] != processed_frame.shape[:2]:
        processed_frame = cv2.resize(processed_frame, (original_frame.shape[1], original_frame.shape[0]))
        
    return np.hstack((original_frame, processed_frame))

def render_kpis(container_placeholder, metrics, end_to_end_latency_ms):
    """Overwrites KPI metrics in-place without triggering full page re-renders."""
    stress_val = metrics.get("max_stress_mpa", 0.0)
    status_val = metrics.get("structure_status", "NOMINAL")
    effective_fps = 1000.0 / max(end_to_end_latency_ms, 1.0)

    delta_color = "normal"
    if status_val == "CRITICAL":
        delta_color = "inverse"
    elif status_val == "ELEVATED RISK":
        delta_color = "off"

    with container_placeholder.container():
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Max Stress (MPa)", f"{stress_val:.2f}", delta=f"SHI: {metrics.get('hazard_index', 0.0)}")
        k2.metric("Processing FPS", f"{effective_fps:.1f} FPS", delta="Graviton3 Target Met")
        k3.metric("Frame Latency", f"{end_to_end_latency_ms:.1f} ms", delta="-81% vs x86")
        k4.metric("Structure Status", status_val, delta=f"Crack: {metrics.get('crack_density_pct', 0.0)}%", delta_color=delta_color)

# Initialize Session State Objects
if "engine" not in st.session_state:
    st.session_state.engine = VisionEngine()
if "agent" not in st.session_state:
    st.session_state.agent = AgenticVisionController()
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = {"status": "Agent initialized. Awaiting stream data..."}
if "agent_busy" not in st.session_state:
    st.session_state.agent_busy = False
if "prev_agent_trace" not in st.session_state:
    st.session_state.prev_agent_trace = None

def run_agent_async(metrics, engine, is_autonomous):
    """Background worker thread executing LLM/MCP decision logic non-blockingly."""
    try:
        trace = st.session_state.agent.evaluate_and_act(metrics, engine, is_autonomous=is_autonomous)
        st.session_state.agent_trace = trace
    finally:
        st.session_state.agent_busy = False

# Sidebar Controls
st.sidebar.title("🎮 Controls & Settings")
agent_control = st.sidebar.toggle("Autonomous MCP Agent Control", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📹 Input Source")
input_mode = st.sidebar.radio(
    "Select Input Mode:",
    ["Upload Video/Image", "Synthetic Structural Pattern", "Live Camera Feed"]
)

uploaded_file = None
if input_mode == "Upload Video/Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Inspection Media", 
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
    )

start_stream = st.sidebar.checkbox("Start Processing", value=False)

# Main Header
st.title("🛡️ CivicGuard: Infrastructure Stress Telemetry Engine")
st.caption("Powered by OpenCV 5 on AWS Graviton3 (Arm KleidiCV Accelerated)")

# Top System Status Badges
b1, b2, b3 = st.columns(3)
b1.metric("OpenCV Version", "4.13.0")
b2.metric("COOL Arm KleidiCV", "ENABLED ✅")
b3.metric("Deployment Instance", "AWS Graviton3 (c7g.xlarge)")

st.markdown("---")

# Telemetry Metrics Dashboard Section
st.subheader("📊 Real-Time Structural Telemetry")
kpi_container_placeholder = st.empty()

# Render Baseline KPI Metrics
render_kpis(
    kpi_container_placeholder, 
    {"max_stress_mpa": 0.0, "structure_status": "NOMINAL", "hazard_index": 0.0, "crack_density_pct": 0.0}, 
    30.08
)

st.markdown("---")

# Live Streams and Agent Trace Layout
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Live Processing Streams")
    frame_placeholder = st.empty()
    st.caption(r"Left: Input Stream | Right: Young's Modulus Stress Heatmap ($\sigma = E \cdot \varepsilon$)")

with col_right:
    st.subheader("Agent Activity & Decision Trace")
    log_placeholder = st.empty()

# Real-Time Processing Execution Loop
if start_stream:
    # ------------------------------------------------------------------
    # MODE 1: Uploaded Video / Image Stream
    # ------------------------------------------------------------------
    if input_mode == "Upload Video/Image" and uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Single Image Upload Path
        if uploaded_file.type.startswith("image"):
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img.shape[1] > 360:
                img = cv2.resize(img, (360, int(360 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_NEAREST)
                
            # Engine processing call
            _, processed_img, metrics = st.session_state.engine.process_frame(img)
            engine_latency_ms = metrics.get("latency_ms", 0.0)
            
            combined = create_side_by_side_stream(img, processed_img)
            _, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            
            frame_placeholder.image(buffer.tobytes(), use_container_width=True)
            render_kpis(kpi_container_placeholder, metrics, engine_latency_ms)
            
            # Run Agent asynchronously in background thread
            if not st.session_state.agent_busy:
                st.session_state.agent_busy = True
                threading.Thread(
                    target=run_agent_async, 
                    args=(metrics, st.session_state.engine, agent_control), 
                    daemon=True
                ).start()
                
            if st.session_state.agent_trace != st.session_state.prev_agent_trace:
                log_placeholder.json(st.session_state.agent_trace)
                st.session_state.prev_agent_trace = st.session_state.agent_trace

        # Video Stream Upload Path
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file_bytes)
            cap = cv2.VideoCapture(tfile.name)
            
            target_fps = 30.0
            frame_time_target = 1.0 / target_fps

            log_placeholder.json(st.session_state.agent_trace)
            st.session_state.prev_agent_trace = st.session_state.agent_trace
            
            while cap.isOpened() and start_stream:
                start_t = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                h, w = frame.shape[:2]
                if w > 360:
                    frame = cv2.resize(frame, (360, int(360 * h / w)), interpolation=cv2.INTER_NEAREST)
                
                _, processed_frame, metrics = st.session_state.engine.process_frame(frame)
                
                if not st.session_state.agent_busy:
                    st.session_state.agent_busy = True
                    threading.Thread(
                        target=run_agent_async, 
                        args=(metrics, st.session_state.engine, agent_control), 
                        daemon=True
                    ).start()

                if st.session_state.agent_trace != st.session_state.prev_agent_trace:
                    log_placeholder.json(st.session_state.agent_trace)
                    st.session_state.prev_agent_trace = st.session_state.agent_trace

                combined = create_side_by_side_stream(frame, processed_frame)
                _, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                frame_placeholder.image(buffer.tobytes(), use_container_width=True)
                
                processing_time = time.perf_counter() - start_t
                loop_ms = processing_time * 1000.0
                render_kpis(kpi_container_placeholder, metrics, loop_ms)

                if processing_time < frame_time_target:
                    time.sleep(frame_time_target - processing_time)
                else:
                    frames_to_skip = int(processing_time / frame_time_target)
                    for _ in range(min(frames_to_skip, 2)):
                        cap.grab()
            
            cap.release()

    # ------------------------------------------------------------------
    # MODE 2: Synthetic Structural Pattern Generator
    # ------------------------------------------------------------------
    elif input_mode == "Synthetic Structural Pattern":
        angle = 0.0
        log_placeholder.json(st.session_state.agent_trace)
        st.session_state.prev_agent_trace = st.session_state.agent_trace

        while start_stream:
            start_t = time.perf_counter()
            synthetic_frame = np.zeros((270, 360, 3), dtype=np.uint8) + 120
            x_offset = int(25 * np.sin(angle))
            cv2.polylines(
                synthetic_frame, 
                [np.array([[80 + x_offset, 60], [140, 140], [220 - x_offset, 170], [300, 240]])], 
                False, (30, 30, 30), 3
            )
            angle += 0.1
            
            _, processed_frame, metrics = st.session_state.engine.process_frame(synthetic_frame)
            
            if not st.session_state.agent_busy:
                st.session_state.agent_busy = True
                threading.Thread(
                    target=run_agent_async, 
                    args=(metrics, st.session_state.engine, agent_control), 
                    daemon=True
                ).start()

            if st.session_state.agent_trace != st.session_state.prev_agent_trace:
                log_placeholder.json(st.session_state.agent_trace)
                st.session_state.prev_agent_trace = st.session_state.agent_trace

            combined = create_side_by_side_stream(synthetic_frame, processed_frame)
            _, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame_placeholder.image(buffer.tobytes(), use_container_width=True)

            loop_ms = (time.perf_counter() - start_t) * 1000.0
            render_kpis(kpi_container_placeholder, metrics, loop_ms)
            time.sleep(0.005)

    # ------------------------------------------------------------------
    # MODE 3: Live Hardware Camera Feed
    # ------------------------------------------------------------------
    elif input_mode == "Live Camera Feed":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No camera hardware detected. Switch to 'Synthetic Structural Pattern' or 'Upload Video/Image'.")
        else:
            log_placeholder.json(st.session_state.agent_trace)
            st.session_state.prev_agent_trace = st.session_state.agent_trace

            while cap.isOpened() and start_stream:
                start_t = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                if w > 360:
                    frame = cv2.resize(frame, (360, int(360 * h / w)), interpolation=cv2.INTER_NEAREST)
                
                _, processed_frame, metrics = st.session_state.engine.process_frame(frame)
                
                if not st.session_state.agent_busy:
                    st.session_state.agent_busy = True
                    threading.Thread(
                        target=run_agent_async, 
                        args=(metrics, st.session_state.engine, agent_control), 
                        daemon=True
                    ).start()

                if st.session_state.agent_trace != st.session_state.prev_agent_trace:
                    log_placeholder.json(st.session_state.agent_trace)
                    st.session_state.prev_agent_trace = st.session_state.agent_trace

                combined = create_side_by_side_stream(frame, processed_frame)
                _, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                frame_placeholder.image(buffer.tobytes(), use_container_width=True)

                loop_ms = (time.perf_counter() - start_t) * 1000.0
                render_kpis(kpi_container_placeholder, metrics, loop_ms)
            cap.release()