import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch
import time
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F

# --- Import our custom OOP pipeline ---
from src import SatelliteColorizer, GlobalSceneNormalizer

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="ISRO PS-10 | IR-to-RGB Framework",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for a sleeker dashboard look
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1E88E5; font-family: 'Helvetica Neue', sans-serif; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=150)
    st.title("Project Control Panel")
    st.markdown("---")
    st.markdown("**Hackathon:** ISRO BAC 2024")
    st.markdown("**Problem Statement:** 10")
    st.markdown("**Task:** Infrared Image Colorization & Enhancement")
    st.markdown("---")
    st.info(
        "💡 **How it works:**\n"
        "1. Upload a raw Landsat 8/9 Thermal IR `.tif` file.\n"
        "2. The Global Scene Normalizer scales the scientific telemetry.\n"
        "3. Our custom Pix2Pix GAN reconstructs high-fidelity RGB textures."
    )

# ==========================================
# 3. MAIN DASHBOARD HEADER
# ==========================================
st.title("🛰️ Geospatial Thermal-to-RGB Enhancement Pipeline")
st.markdown(
    "Transform monochrome, low-visibility satellite thermal infrared telemetry into "
    "high-resolution, structurally accurate visible-spectrum imagery using Generative Adversarial Networks."
)

# ==========================================
# 4. LOAD AI PIPELINE (Cached for speed)
# ==========================================
@st.cache_data
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    config = load_config()
    try:
        return SatelliteColorizer(
            config=config
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner("Initializing Deep Learning Engine..."):
    pipeline = load_pipeline()

# ==========================================
# 5. USER INTERFACE & UPLOAD LOGIC
# ==========================================
if pipeline:
    uploaded_file = st.file_uploader(
        "Upload a single-band Thermal IR GeoTIFF (.tif) file to begin inference", 
        type=["tif", "tiff"]
    )

    if uploaded_file is not None:
        try:
            st.markdown("---")
            st.subheader("Inference Results")
            
            # Start timer for inference metric
            start_time = time.time()
            
            # Read file into memory (avoids saving to disk)
            with MemoryFile(uploaded_file.read()) as memfile:
                with memfile.open() as src:
                    ir_array = src.read(1).astype(np.float32)
            
            # --- PREPROCESSING ---
            
            normalizer = GlobalSceneNormalizer()
            norm_array = normalizer.fit_transform(ir_array)
            
            # Reshape for PyTorch: (Batch=1, Channel=1, Height, Width)
            input_tensor = torch.tensor(norm_array).unsqueeze(0).unsqueeze(0)
            
            # CRITICAL FIX: Resize the tensor to 256x256 to perfectly align with U-Net skip connections
            input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
            
            # --- INFERENCE ---
            with st.spinner("Neural Network painting synthetic features..."):
                predicted_rgb = pipeline.predict(input_tensor)
            
            inference_time = time.time() - start_time
            
            # --- DISPLAY DASHBOARD ---
            # Top row: Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(label="Input Resolution", value=f"{ir_array.shape[1]}x{ir_array.shape[0]} px")
            col_m2.metric(label="Global Array Range", value=f"{normalizer.vmin:.2f} - {normalizer.vmax:.2f}")
            col_m3.metric(label="Inference Latency", value=f"{inference_time:.3f} sec")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Bottom row: Images
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("#### 📡 Raw Telemetry (Input)")
                fig_in, ax_in = plt.subplots()
                # Magma colormap highlights thermal differences beautifully
                cax = ax_in.imshow(ir_array, cmap='magma')
                ax_in.axis('off')
                fig_in.colorbar(cax, orientation='horizontal', pad=0.05, aspect=40)
                st.pyplot(fig_in)
                st.caption("Values mapped using Magma colormap for visual distinction.")

            with col_img2:
                st.markdown("#### 🌍 AI Reconstruction (Output)")
                fig_out, ax_out = plt.subplots()
                ax_out.imshow(predicted_rgb)
                ax_out.axis('off')
                st.pyplot(fig_out)
                st.caption("Synthesized visible-spectrum features preserving structural integrity.")
                
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.info("Please ensure the uploaded file is a valid 1-channel GeoTIFF.")