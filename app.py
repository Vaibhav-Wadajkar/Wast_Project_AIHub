import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import urllib.request
import json
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="WasteSight AI Pro",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# MODERN ECO-INDUSTRIAL GLASS THEME (CSS)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Base Overrides */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #040905 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #e2f0e5 !important;
}

/* Subtle Ambient Background Glows */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: -10%; left: -10%; width: 50vw; height: 50vh;
    background: radial-gradient(circle, rgba(74, 173, 91, 0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Hide standard Streamlit branding elements but keep header transparent for sidebar toggle */
footer, [data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Card & Glassmorphic Container Panels */
.glass-panel {
    background: rgba(12, 24, 15, 0.65);
    border: 1px solid rgba(74, 173, 91, 0.15);
    border-radius: 18px;
    padding: 1.75rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.hero-panel {
    background: linear-gradient(135deg, rgba(16, 36, 20, 0.8) 0%, rgba(6, 14, 7, 0.95) 100%);
    border: 1px solid rgba(74, 173, 91, 0.3);
    border-radius: 24px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px 0 rgba(74, 173, 91, 0.05);
}

/* Custom Text Typographies */
.hero-title {
    font-size: 3rem !important;
    font-weight: 800 !important;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 40%, #76e287 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #76e287;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Custom List Styles */
.bullet-list {
    list-style: none;
    padding-left: 0;
    margin: 0;
}
.bullet-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    font-size: 0.88rem;
    color: #b3cbba;
    margin-bottom: 0.6rem;
    line-height: 1.4;
}
.bullet-dot {
    color: #76e287;
    font-weight: bold;
}

/* Streamlit Native Component Stylings Overrides */
[data-testid="stFileUploader"] {
    background: rgba(16, 32, 20, 0.3) !important;
    border: 1.5px dashed rgba(74, 173, 91, 0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #76e287 !important;
}

/* Metric styling adjustments */
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #8fae96 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# RESOURCE LOADERS (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Initializing Neural Engine...")
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    # Optional: move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

@st.cache_data(show_spinner=False)
def load_labels():
    import os
    local_path = "imagenet-simple-labels.json"
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as f:
            return json.load(f)
    except Exception:
        # Fallback minimal labels
        st.warning("Could not fetch labels from GitHub, using minimal fallback.")
        return ["unknown"] * 1000  # dummy, but better than crash

# ImageNet Preprocessing Transformations Pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==========================================
# WASTE CLASSIFICATION ENGINE LOGIC
# ==========================================
WASTE_MAP = {
    # Organic / Compostable
    "banana": "organic", "apple": "organic", "orange": "organic", "lemon": "organic",
    "pineapple": "organic", "strawberry": "organic", "cucumber": "organic", "lettuce": "organic",
    "broccoli": "organic", "artichoke": "organic", "cauliflower": "organic", "mushroom": "organic",
    "pizza": "organic", "bagel": "organic", "pretzel": "organic", "bread": "organic",
    "corn": "organic", "cabbage": "organic", "zucchini": "organic", "squash": "organic", "fig": "organic",
    # Dry Recyclables
    "plastic bottle": "recyclable", "water bottle": "recyclable", "can": "recyclable",
    "tin can": "recyclable", "aluminum can": "recyclable", "newspaper": "recyclable",
    "paper": "recyclable", "cardboard": "recyclable", "envelope": "recyclable",
    "box": "recyclable", "jar": "recyclable", "bottle": "recyclable",
    # Hazardous Tech & Medical E-Waste
    "battery": "hazardous", "mobile phone": "hazardous", "cell phone": "hazardous",
    "remote control": "hazardous", "laptop": "hazardous", "computer": "hazardous",
    "television": "hazardous", "monitor": "hazardous", "printer": "hazardous",
    "syringe": "hazardous", "pill bottle": "hazardous",
}

GUIDE = {
    "organic": {
        "icon": "🌿", "label": "Organic Waste", "status": "complete",
        "bin": "🟢 Green Bin (Compost Core)",
        "tips": [
            "Strip away all synthetic stickers, tags, or plastic bands before disposal.",
            "Raw fruit peels and organic food residuals process cleanly into rich agriculture soil.",
            "Avoid introducing heavily processed dairy or high-oil cooking sauces into regular garden bins."
        ]
    },
    "recyclable": {
        "icon": "♻️", "label": "Recyclable Material", "status": "complete",
        "bin": "🔵 Blue Bin (Dry Recyclables)",
        "tips": [
            "Thoroughly rinse out containers to eliminate liquid foods or raw sugar contamination.",
            "Break down, collapse, and flatten shipping cartons to minimize storage friction.",
            "Always inspect regional guidelines: soft plastics/films usually call for independent channels."
        ]
    },
    "hazardous": {
        "icon": "⚡", "label": "Hazardous Material", "status": "error",
        "bin": "🔴 Specialized Hazardous Facility Drop-off",
        "tips": [
            "Strictly prohibited from landfills; do not toss into common residential garbage systems.",
            "Preserve local water tables by dropping e-waste at authorized urban recovery stations.",
            "Take advantage of consumer retail drop bins for home cell batteries and small gadgets."
        ]
    },
    "unknown": {
        "icon": "🔍", "label": "Unidentified Object", "status": "complete",
        "bin": "🟡 General Household Waste / Verify Locally",
        "tips": [
            "Isolate the single target item against a cleanly lit, plain, minimalist background.",
            "When validation remains uncertain, access your municipal sanitary portal for definitive cataloging."
        ]
    }
}

def classify_waste(predictions_list) -> str:
    """
    Checks top predictions for any waste keyword match.
    Returns the category of the first match, or 'unknown' if none found.
    """
    for label, _ in predictions_list:
        cleaned_label = label.lower()
        for pattern, category in WASTE_MAP.items():
            if pattern in cleaned_label:
                return category
    return "unknown"

# ==========================================
# SIDEBAR NAVIGATION & INFO PANEL
# ==========================================
with st.sidebar:
    st.markdown('<div style="padding: 1rem 0; text-align: center;"><h2 style="margin:0; color:#76e287; font-weight:800; letter-spacing:-0.04em;">♻️ WasteSight Pro</h2><p style="font-size:0.8rem; color:#8fae96; margin:0; text-transform:uppercase; letter-spacing:0.1em;">Intelligent Sorting System</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div class="section-label">Workflow Execution</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Supply structural asset photograph"),
        ("2", "MobileNetV2 runs safe local inference"),
        ("3", "Instantly route item to correct bin")
    ]
    for num, text in steps:
        st.markdown(f'<div style="display:flex; gap:0.75rem; margin-bottom:0.75rem;"><div style="background:rgba(118,226,135,0.15); color:#76e287; border:1.5px solid rgba(118,226,135,0.3); border-radius:6px; min-width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700;">{num}</div><div style="font-size:0.85rem; color:#b3cbba; line-height:1.4;">{text}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:2rem;">Core Categories</div>', unsafe_allow_html=True)
    categories = [
        ("🌿 Organic", "Compostable items"),
        ("♻️ Recyclable", "Paper, clean metals, rigid plastic"),
        ("⚡ Hazardous", "E-waste, chemical items, batteries")
    ]
    for cat_title, cat_desc in categories:
        st.markdown(f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:0.5rem 0.75rem; border-radius:10px; margin-bottom:0.5rem;"><div style="font-size:0.85rem; font-weight:600; color:#e2f0e5;">{cat_title}</div><div style="font-size:0.72rem; color:#8fae96;">{cat_desc}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📊 Technical Specifications"):
        st.markdown("""
        <div style="font-size:0.8rem; color:#b3cbba; line-height:1.5;">
        <b>Architecture:</b> MobileNetV2<br>
        <b>Weights:</b> ImageNet Pre-trained<br>
        <b>Privacy:</b> 100% Client-Side Sandbox<br>
        <b>Framework:</b> PyTorch x Streamlit
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MAIN INTERFACE HEADER HERO
# ==========================================
st.markdown("""
<div class="hero-panel">
    <div style="position:absolute; right:2rem; top:1rem; font-size:8rem; opacity:0.04; pointer-events:none;">🌿</div>
    <div style="display:inline-flex; align-items:center; gap:0.5rem; background:rgba(118,226,135,0.12); border:1px solid rgba(118,226,135,0.25); padding:0.25rem 0.75rem; border-radius:50px; font-size:0.7rem; font-weight:700; color:#76e287; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:1rem;">
        <span style="width:6px; height:6px; background:#76e287; border-radius:50%;"></span> Edge Engine Active
    </div>
    <div class="hero-title">Optimize Recycling Accuracy.<br>Eliminate Contamination.</div>
    <p style="color:#b3cbba; max-width:650px; margin:0; font-size:0.95rem; line-height:1.6;">
        Upload a picture of any household or industrial item. The on-device computer vision engine determines the material type and returns instant, compliant disposal directions.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN TWO-COLUMN VIEWPORT
# ==========================================
col_input, col_output = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: INPUT CONTROLS ---
with col_input:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📸 Scan Item</div>', unsafe_allow_html=True)
    
    pil_image = None
    uploaded_file = st.session_state.get("uploaded_file", None)

    if uploaded_file is None:
        uploaded_file = st.file_uploader(
            "Drop image or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="file_uploader_key"
        )
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            st.rerun()
            
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image)
            
            # Metadata & Remove Button Grid
            col_meta, col_btn = st.columns([3, 2])
            with col_meta:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:0.6rem 1rem; font-size:0.8rem; height:40px; display:flex; align-items:center; overflow:hidden;">
                    <span style="color:#8fae96; text-overflow:ellipsis; overflow:hidden; white-space:nowrap; max-width:180px;">📄 {uploaded_file.name}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                # Custom styled Streamlit button to clear and rerun
                if st.button("✖ Remove", use_container_width=True):
                    st.session_state["uploaded_file"] = None
                    st.rerun()
        except Exception as e:
            st.error(f"Failed to open image: {e}")
            st.session_state["uploaded_file"] = None
            pil_image = None
    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px dashed rgba(255,255,255,0.08); border-radius:14px; min-height:220px; padding:2rem; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem; opacity:0.4;">📸</div>
            <div style="font-size:0.9rem; font-weight:600; color:#e2f0e5; margin-bottom:0.25rem;">Waiting for Source Image</div>
            <div style="font-size:0.78rem; color:#8fae96; max-width:240px;">Provide an item image to trigger real-time neural network inference.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Informational Value Add Card
    st.markdown("""
    <div class="glass-panel" style="background:rgba(118,226,135,0.02);">
        <div style="display:flex; gap:1rem; align-items:flex-start;">
            <div style="font-size:1.5rem;">💡</div>
            <div>
                <div style="font-size:0.88rem; font-weight:700; color:#e2f0e5; margin-bottom:0.25rem;">Why Cross-Contamination Matters</div>
                <div style="font-size:0.8rem; color:#8fae96; line-height:1.5; margin:0;">
                    Tossing a single un-rinsed sauce jar or alkaline battery into general recycling batches can ruin thousands of pounds of clean material, routing entire truckloads directly to carbon-heavy landfills.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- RIGHT COLUMN: ANALYSIS OUTPUTS ---
with col_output:
    st.markdown('<div class="glass-panel" style="min-height: 100%;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🤖 Analysis & Metrics</div>', unsafe_allow_html=True)
    
    if not uploaded_file or pil_image is None:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:350px; text-align:center;">
            <div style="font-size:3rem; margin-bottom:0.75rem; opacity:0.2;">📊</div>
            <div style="font-size:0.95rem; font-weight:600; color:#8fae96;">Awaiting Stream Input</div>
            <div style="font-size:0.8rem; color:#5c7d5e; max-width:220px; margin-top:0.25rem;">Telemetry and local model logs populate dynamically here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Load model and device
        nn_model, device = load_model()
        class_labels = load_labels()

        # Prepare input tensor
        input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            t_start = time.time()
            logits = nn_model(input_tensor)
            inference_delta_ms = (time.time() - t_start) * 1000
            probs = torch.nn.functional.softmax(logits[0], dim=0)

        # Gather Top 3 Classes
        top3_probs, top3_indices = torch.topk(probs, 3)
        
        top_predictions = []
        for index in range(3):
            p_val = top3_probs[index].item()
            l_str = class_labels[top3_indices[index].item()]
            top_predictions.append((l_str, p_val))

        primary_class_str, primary_confidence = top_predictions[0]
        
        # Classify waste using all top predictions
        mapped_cat = classify_waste(top_predictions)
        guide_meta = GUIDE[mapped_cat]

        # Dynamic Status Callout Header
        with st.status(
            label=f"Analysis Finished — Structured Object Routed to {guide_meta['label']}",
            expanded=True,
            state=guide_meta["status"]
        ):
            st.write(f"Primary Tensor Check: Recognized pattern resembling '{primary_class_str}' safely mapped.")
        
        st.markdown('<div style="margin-top:1.5rem; margin-bottom:1.5rem;">', unsafe_allow_html=True)
        st.markdown(f"### {guide_meta['icon']} {guide_meta['bin']}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Native Progress Metrics Grid for Predictions
        st.markdown('<p style="font-size:0.8rem; font-weight:700; color:#8fae96; text-transform:uppercase; margin-bottom:0.75rem;">Model Class Confidence</p>', unsafe_allow_html=True)
        for ranked_label, confidence_score in top_predictions:
            col_lbl, col_bar = st.columns([2, 3])
            with col_lbl:
                st.markdown(f"<span style='font-size:0.85rem; color:#e2f0e5; text-transform:capitalize;'>{ranked_label}</span>", unsafe_allow_html=True)
            with col_bar:
                st.progress(confidence_score, text=f"{round(confidence_score * 100, 1)}%")

        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

        # Render Core Disposal Tips via styled lists
        st.markdown('<p style="font-size:0.8rem; font-weight:700; color:#8fae96; text-transform:uppercase; margin-bottom:0.75rem;">Critical Disposal Guide</p>', unsafe_allow_html=True)
        st.markdown('<ul class="bullet-list">', unsafe_allow_html=True)
        for current_tip in guide_meta["tips"]:
            st.markdown(f'<li class="bullet-item"><span class="bullet-dot">✓</span><span>{current_tip}</span></li>', unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)

        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

        # Performance Infrastructure Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="Inference Latency", value=f"{round(inference_delta_ms, 1)} ms")
        with col_m2:
            st.metric(label="Primary Certainty", value=f"{round(primary_confidence * 100, 1)}%")
        with col_m3:
            st.metric(label="Data Privacy", value="100% Edge")

    st.markdown('</div>', unsafe_allow_html=True)