import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
from PIL import Image
from yolov8_surgical_monitor import SurgicalMonitor

# --- Page configuration ---
st.set_page_config(
    page_title="Surgical Vision Dashboard", 
    page_icon="👁️\u200d🗨️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for aesthetic, 3-panel layout UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Subtle gradient text for main title */
    .main-title {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
        text-align: center;
    }

    /* Panels / Cards */
    .panel-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .panel-header {
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
        margin-bottom: 12px;
        font-size: 1.2rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.4);
    }
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: #0f172a;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #38bdf8;
        background-color: rgba(56, 189, 248, 0.05);
    }
    
    /* Metrics Override */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background: #0f172a;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 12px;
        text-align: center;
    }
    
    /* Sliders and Toggles padding */
    .stSlider, .stCheckbox {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State for Timer ---
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()

# --- Header Section ---
st.markdown('<h1 class="main-title">👁️\u200d🗨️ Surgical Vision_v17</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#94a3b8; margin-bottom: 30px;'>Next-Gen Inattentional Blindness Monitor</p>", unsafe_allow_html=True)

@st.cache_resource
def load_monitor():
    from pathlib import Path
    base_dir = Path(__file__).parent.absolute()
    
    # Priority paths for different environments
    local_path = base_dir / 'best.pt'
    runs_path = base_dir / 'runs' / 'detect' / 'surgical_detector_v17' / 'weights' / 'best.pt'
    
    if local_path.exists():
        model_path = str(local_path)
        status = "Specialist (L-Path)"
    elif runs_path.exists():
        model_path = str(runs_path)
        status = "Specialist (R-Path)"
    else:
        model_path = 'yolov8n.pt'
        status = "Fallback: Generic YOLOv8n"
    
    # Ensure diagnostics also check the absolute path
    st.session_state['abs_best_pt'] = local_path.exists()
    
    return SurgicalMonitor(model_path=model_path, confidence_threshold=0.15), status

monitor, model_status = load_monitor()

# Define global state text for badge
badge_state = "🟢 Model Ready"


# --- 3-PANEL LAYOUT STRUCTURE ---
# Left (Controls 1.0), Center (Feed 3.0), Right (Analytics 1.0)
col_left, col_center, col_right = st.columns([1.0, 3.0, 1.0], gap="medium")


# ==========================================
# LEFT PANEL: CONTROLS
# ==========================================
with col_left:
    st.markdown("""
        <div class="panel-container">
            <div class="panel-header">🎛️ Monitoring Controls</div>
    """, unsafe_allow_html=True)
    
    mode = st.radio("Input Source", ["🖼️ Image", "🎬 Video", "🎥 Live Camera"])
    st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)
    
    st.markdown("#### Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.15, step=0.05, 
                               help="Minimum confidence score required to display a bounding box.")
    
    show_focus = st.toggle("Show Focus Zone Overlay", value=True, 
                           help="Display the green primary focus zone on the feed.")
    
    alert_margin = st.slider("Alert Sensitivity (Focus Margin)", min_value=0.0, max_value=0.5, value=0.2, step=0.05,
                             help="Adjust the size of the central focus zone. Lower means a larger focus area.")
    
    show_generic = st.toggle("Show Generic Objects", value=False, 
                             help="Display non-surgical objects like people or bottles.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**Engine:** {model_status}")

    # --- System Diagnostics ---
    with st.expander("🛠️ System Health"):
        st.write(f"**Python:** {os.sys.version.split(' ')[0]}")
        st.write(f"**Working Dir:** `{os.getcwd()}`")
        
        # Check for critical files using the robust path
        from pathlib import Path
        best_found = (Path(__file__).parent / 'best.pt').exists()
        st.write(f"**best.pt found:** {'✅' if best_found else '❌'}")
        
        if st.checkbox("List all files"):
            st.code("\n".join(os.listdir('.')))

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# RIGHT PANEL: LIVE ANALYTICS (Placeholders)
# ==========================================
with col_right:
    st.markdown("""
        <div class="panel-container">
            <div class="panel-header">📈 Live Analytics</div>
    """, unsafe_allow_html=True)
    
    # We create empty placeholders so we can update them dynamically inside the Center Panel's loop
    metric_fps = st.empty()
    metric_timer = st.empty()
    metric_tools = st.empty()
    metric_alerts = st.empty()
    metric_conf = st.empty()
    
    def update_metrics(fps, tools, alerts, avg_conf):
        elapsed = int(time.time() - st.session_state.session_start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        with metric_fps.container():
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase;">Stream FPS</div>
                <div style="color: #38bdf8; font-size: 2.2rem; font-weight: bold;">{fps:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_timer.container():
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase;">Session Time</div>
                <div style="color: #cbd5e1; font-size: 1.8rem; font-weight: bold;">{time_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_tools.container():
            st.markdown(f"""
            <div class="metric-card" style="border-color: #4ade80;">
                <div style="color: #4ade80; font-size: 0.95rem; text-transform: uppercase;">Surgical Tools</div>
                <div style="color: #4ade80; font-size: 2.2rem; font-weight: bold;">{tools}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_alerts.container():
            alert_color = "#f87171" if alerts > 0 else "#94a3b8"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {alert_color};">
                <div style="color: {alert_color}; font-size: 0.95rem; text-transform: uppercase;">Outside Alerts</div>
                <div style="color: {alert_color}; font-size: 2.2rem; font-weight: bold;">{alerts}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_conf.container():
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase;">Avg Confidence</div>
                <div style="color: #38bdf8; font-size: 1.8rem; font-weight: bold;">{avg_conf:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
            
    # Initialize empty state
    update_metrics(0.0, 0, 0, 0.0)
    
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# CENTER PANEL: LIVE SURGICAL FEED
# ==========================================
mode_clean = mode.split(" ")[1]

with col_center:
    # --- Top Right Status Badge Logic ---
    # We display it right above the central panel for highest visibility
    if mode_clean == "Image" and st.session_state.get('img_processing', False):
        badge_state = "🟡 Processing"
    elif mode_clean == "Video" and st.session_state.get('vid_processing', False):
        badge_state = "🟡 Analyzing"
    elif mode_clean == "Live" and st.session_state.get('webcam_active', False):
        badge_state = "🟢 Live Monitoring"
    else:
        badge_state = "🟢 System Ready"
        
    st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: #cbd5e1; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                {badge_state}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="panel-container" style="padding: 1rem;">
    """, unsafe_allow_html=True)
    
    if mode_clean == "Image":
        uploaded_file = st.file_uploader("Upload a surgical frame...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("Analyzing Frame..."):
                st.session_state.img_processing = True
                start_time = time.time()
                processed_image, stats = monitor.process_frame(
                    image, 
                    custom_conf=conf_threshold, 
                    alert_margin=alert_margin, 
                    show_focus_zone=show_focus,
                    show_generic=show_generic
                )
                fps = 1.0 / (time.time() - start_time)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Update right panel analytics!
            update_metrics(fps, stats["instrument_count"], stats["outside_alerts"], stats["avg_confidence"])
            
            st.image(processed_image, use_column_width=True, clamp=True)
            st.session_state.img_processing = False
        else:
            st.markdown("""
            <div style='height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #0f172a; border-radius: 8px; border: 2px dashed #334155; position: relative; overflow: hidden;'>
                <div style='position: absolute; font-size: 15rem; color: rgba(51, 65, 85, 0.1); user-select: none;'>🖼️</div>
                <h3 style='color: #94a3b8 !important; margin-bottom: 5px; z-index: 1;'>Awaiting Surgical Frame</h3>
                <p style='color: #64748b; font-size: 0.95rem; text-align: center; max-width: 80%; z-index: 1;'>Upload an image above to begin real-time monitoring and anomaly detection.</p>
                <div style='background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; margin-top: 15px; font-weight: 600; z-index: 1;'>
                    Supported formats: JPG, JPEG, PNG
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif mode_clean == "Video":
        uploaded_file = st.file_uploader("Upload procedural video clip...", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stop_btn = st.button("⏹️ Stop Playback")
            
            stframe = st.empty()
            
            st.session_state.vid_processing = True
            
            # For FPS calculation
            prev_time = time.time()
            
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                processed_frame, stats = monitor.process_frame(
                    frame, 
                    custom_conf=conf_threshold, 
                    alert_margin=alert_margin, 
                    show_focus_zone=show_focus,
                    show_generic=show_generic
                )
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                update_metrics(fps, stats["instrument_count"], stats["outside_alerts"], stats["avg_confidence"])
                
            cap.release()
            st.session_state.vid_processing = False
            try:
                os.remove(tfile.name)
            except:
                pass
        else:
            st.markdown("""
            <div style='height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #0f172a; border-radius: 8px; border: 2px dashed #334155; position: relative; overflow: hidden;'>
                <div style='position: absolute; font-size: 15rem; color: rgba(51, 65, 85, 0.1); user-select: none;'>🎬</div>
                <h3 style='color: #94a3b8 !important; margin-bottom: 5px; z-index: 1;'>Awaiting Procedural Video</h3>
                <p style='color: #64748b; font-size: 0.95rem; text-align: center; max-width: 80%; z-index: 1;'>Upload a video clip to run continuous temporal analysis across the focus zone.</p>
                <div style='background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; margin-top: 15px; font-weight: 600; z-index: 1;'>
                    Supported formats: MP4, AVI, MOV
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif mode_clean == "Live":
        st.markdown("<h4 style='text-align:center; padding-top: 10px;'>Camera Optics</h4>", unsafe_allow_html=True)
        run_webcam = st.toggle("🟢 Activate Live Feed", value=False)
        st.session_state.webcam_active = run_webcam
        stframe = st.empty()
        
        if not run_webcam:
            stframe.markdown("""
            <div style='height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #0f172a; border-radius: 8px; border: 2px dashed #334155; position: relative; overflow: hidden;'>
                <div style='position: absolute; font-size: 15rem; color: rgba(51, 65, 85, 0.1); user-select: none;'>🎥</div>
                <h3 style='color: #94a3b8 !important; margin-bottom: 5px; z-index: 1;'>Camera Optics Offline</h3>
                <p style='color: #64748b; font-size: 0.95rem; text-align: center; max-width: 80%; z-index: 1;'>Toggle "Activate Live Feed" above to stream directly from your device camera.</p>
                <div style='background: rgba(248, 113, 113, 0.1); color: #f87171; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; margin-top: 15px; font-weight: 600; z-index: 1;'>
                    Status: Standby
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        if run_webcam:
            cap = cv2.VideoCapture(0)
            prev_time = time.time()
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Hardware Warning: Unable to acquire camera feed.")
                    break
                    
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                processed_frame, stats = monitor.process_frame(
                    frame, 
                    custom_conf=conf_threshold, 
                    alert_margin=alert_margin, 
                    show_focus_zone=show_focus,
                    show_generic=show_generic
                )
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                update_metrics(fps, stats["instrument_count"], stats["outside_alerts"], stats["avg_confidence"])
                
                stframe.image(processed_frame, channels="RGB", use_column_width=True)
                
            cap.release()
            st.session_state.webcam_active = False

    st.markdown("</div>", unsafe_allow_html=True)


