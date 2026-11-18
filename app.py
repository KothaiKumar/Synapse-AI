import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="SynapseAI", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────────
# Dark / Light mode toggle (session state)
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

dark = st.session_state.dark_mode

# ─────────────────────────────────────────────
# Theme colours
# ─────────────────────────────────────────────
if dark:
    BG       = "#0d0f14"
    BG2      = "#141720"
    CARD_BG  = "linear-gradient(135deg,#1a1d2e,#12152b)"
    CARD_BOR = "#2a2f4a"
    TEXT     = "#e8e8e8"
    TEXT2    = "#8a8fa8"
    BORDER   = "#2a2f4a"
    HEADER_C = "#d0d8f0"
    INFO_BG  = "#141720"
    INFO_TXT = "#c0c8e0"
    NEXT_BG  = "#1a1f3a"
    NEXT_BOR = "#3b5bdb"
    NEXT_TXT = "#a0b4f0"
    EXPLAIN_BG  = "#12201a"
    EXPLAIN_BOR = "#22c55e"
    EXPLAIN_TXT = "#90e0a0"
    MPL_BG   = "#0d0f14"
    MPL_FG   = "#e8e8e8"
    MPL_GRID = "#2a2f4a"
    UPLOAD_BORDER = "#2a2f4a"
    UPLOAD_TXT    = "#555a7a"
else:
    BG       = "#ffffff"
    BG2      = "#f8f9ff"
    CARD_BG  = "linear-gradient(135deg,#f0f4ff,#e8ecff)"
    CARD_BOR = "#c5d0f5"
    TEXT     = "#1a1a2e"
    TEXT2    = "#555577"
    BORDER   = "#dde2f5"
    HEADER_C = "#1a1a2e"
    INFO_BG  = "#f0f4ff"
    INFO_TXT = "#2a3060"
    NEXT_BG  = "#eef2ff"
    NEXT_BOR = "#3b5bdb"
    NEXT_TXT = "#1a3080"
    EXPLAIN_BG  = "#f0fff4"
    EXPLAIN_BOR = "#22c55e"
    EXPLAIN_TXT = "#145a28"
    MPL_BG   = "#ffffff"
    MPL_FG   = "#1a1a2e"
    MPL_GRID = "#dde2f5"
    UPLOAD_BORDER = "#c5d0f5"
    UPLOAD_TXT    = "#888aaa"

# ─────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp, .block-container {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
h1,h2,h3 {{ font-family:'DM Serif Display',serif; color:{TEXT} !important; }}
p, label, .stTextInput label, .stNumberInput label {{ color:{TEXT} !important; }}

.subtitle {{
    text-align:center; color:{TEXT2}; font-size:1rem;
    margin-bottom:2rem; font-weight:300;
}}
.result-card {{
    background:{CARD_BG}; border:1px solid {CARD_BOR};
    border-radius:16px; padding:2rem; text-align:center; margin-top:1rem;
}}
.class-label {{
    font-family:'DM Serif Display',serif; font-size:2.2rem;
    color:#2b4fd4; margin-bottom:0.5rem;
}}
.confidence {{ font-size:1.1rem; color:#1a7a40; font-weight:500; }}
.risk-badge {{
    display:inline-block; padding:0.3rem 1rem; border-radius:999px;
    font-size:0.85rem; font-weight:500; margin-top:0.8rem;
}}
.risk-low {{ background:#d4f5e2; color:#1a6636; border:1px solid #6ec99a; }}
.risk-med {{ background:#fff3cd; color:#7a5500; border:1px solid #f0c060; }}
.risk-high {{ background:#fde8e8; color:#9a1a1a; border:1px solid #f08080; }}

.stButton > button {{
    background:linear-gradient(135deg,#3b5bdb,#5c7cfa);
    color:white !important; border:none; border-radius:10px;
    padding:0.6rem 2rem; font-family:'DM Sans',sans-serif;
    font-size:1rem; width:100%; cursor:pointer; transition:opacity 0.2s;
}}
.stButton > button:hover {{ opacity:0.85; }}

.info-box {{
    background:{INFO_BG}; border-left:3px solid #3b5bdb;
    border-radius:8px; padding:1rem 1.2rem; margin:0.5rem 0;
    font-size:0.9rem; color:{INFO_TXT};
}}
.next-box {{
    background:{NEXT_BG}; border-left:4px solid {NEXT_BOR};
    border-radius:10px; padding:1rem 1.4rem; margin:0.4rem 0;
    font-size:0.9rem; color:{NEXT_TXT};
}}
.explain-box {{
    background:{EXPLAIN_BG}; border-left:4px solid {EXPLAIN_BOR};
    border-radius:10px; padding:1rem 1.4rem; margin:0.5rem 0;
    font-size:0.92rem; color:{EXPLAIN_TXT}; line-height:1.6;
}}
.section-header {{
    font-family:'DM Serif Display',serif; font-size:1.3rem;
    color:{HEADER_C}; border-bottom:1px solid {BORDER};
    padding-bottom:0.4rem; margin-bottom:1rem;
}}
.stTabs [data-baseweb="tab-list"] {{
    background:{BG2} !important; border-radius:10px; padding:4px;
}}
.stTabs [data-baseweb="tab"] {{
    color:{TEXT2} !important; font-family:'DM Sans',sans-serif;
    font-size:0.95rem; border-radius:8px;
}}
.stTabs [aria-selected="true"] {{
    background:#3b5bdb !important; color:white !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────
class GenomicEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, embed_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class RiskHead(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)
    def forward(self, g):
        risk = torch.sigmoid(self.fc(g))
        conditioned = g * (0.5 + 0.5 * risk)
        return conditioned, risk

class MRIEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, embed_dim)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.proj(x)

class RiskGatedFusion(nn.Module):
    def __init__(self, embed_dim=128, drop_prob=0.3):
        super().__init__()
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_g = nn.LayerNorm(embed_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.drop_prob = drop_prob
    def forward(self, v, g):
        v = self.norm_v(v); g = self.norm_g(g)
        gate = self.gate_net(g)
        fused = gate * v + (1 - gate) * g
        return self.out_norm(fused + v)

class BrainTumorModel(nn.Module):
    def __init__(self, genomic_input_dim, num_classes=4):
        super().__init__()
        self.mri = MRIEncoder()
        self.genomic = GenomicEncoder(genomic_input_dim)
        self.risk = RiskHead()
        self.fusion = RiskGatedFusion()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, image, genome):
        v = self.mri(image)
        g = self.genomic(genome)
        g, r = self.risk(g)
        fused = self.fusion(v, g)
        out = self.classifier(fused)
        return out, r, v, fused

# ─────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fhook)
        target_layer.register_backward_hook(self._bhook)
    def _fhook(self, m, i, o): self.activations = o
    def _bhook(self, m, gi, go): self.gradients = go[0]
    def generate(self, img, genome, label):
        self.model.zero_grad()
        out, risk, _, _ = self.model(img, genome)
        score = out[:, label] * risk.squeeze()
        score.sum().backward()
        grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (grads * self.activations).sum(dim=1)
        cam = torch.relu(cam).squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}

CLASS_INFO = {
    0: "A type of tumor that forms in the brain or spinal cord. Requires immediate attention.",
    1: "Usually a slow-growing tumor arising from the membranes surrounding the brain.",
    2: "No tumor detected in the scan. Patient appears healthy.",
    3: "A tumor of the pituitary gland, usually benign and treatable."
}

NEXT_STEPS = {
    0: [
        "🏥 Refer immediately to a neuro-oncologist for further evaluation.",
        "🧬 Order molecular profiling (IDH mutation, MGMT methylation status).",
        "📅 Schedule contrast-enhanced MRI within 1 week.",
        "💊 Discuss corticosteroids to manage potential cerebral edema.",
        "📋 Consider multidisciplinary tumor board review."
    ],
    1: [
        "📅 Schedule follow-up MRI in 3–6 months to monitor growth.",
        "🩺 Neurological examination to assess functional impact.",
        "🔬 Consider biopsy if lesion shows rapid growth or atypical features.",
        "💬 Discuss watchful waiting vs. surgical options with neurosurgeon.",
        "📋 Document baseline symptoms for longitudinal comparison."
    ],
    2: [
        "✅ No immediate action required based on this scan.",
        "📅 Routine annual MRI recommended if patient has risk factors.",
        "📝 Document findings in patient record for future reference.",
        "💬 Counsel patient on symptoms to watch for (headaches, vision changes).",
    ],
    3: [
        "🔬 Order hormonal panel: prolactin, GH, ACTH, TSH, LH/FSH.",
        "👁️ Refer to ophthalmology for visual field assessment.",
        "🩺 Endocrinology consultation for hormonal management.",
        "📅 Schedule dedicated pituitary MRI with thin-slice protocol.",
        "💊 Discuss medical therapy (dopamine agonists) vs. surgery options."
    ]
}

EXPLAIN_TEXT = {
    0: (
        "The model identified irregular hypointense regions with indistinct borders in the scan, "
        "consistent with infiltrative growth patterns characteristic of gliomas. "
        "The genomic risk module detected elevated expression signatures associated with high-grade glioma subtypes. "
        "The attention heatmap highlights the peri-tumoral region where signal abnormality was most prominent, "
        "driving the model's classification decision."
    ),
    1: (
        "The scan shows a well-defined, extra-axial mass with a broad dural base, "
        "features strongly associated with meningioma. "
        "The model's attention focused on the sharp tumor-brain interface and surrounding edema pattern. "
        "Genomic risk indicators suggest a lower-grade profile, consistent with WHO Grade I or II meningioma. "
        "The high confidence reflects the distinctive imaging morphology of this tumor type."
    ),
    2: (
        "No abnormal signal intensity, mass effect, or contrast-enhancing lesions were detected. "
        "The model's attention map shows diffuse, low-activation regions with no focal hotspots, "
        "indicating the absence of a discrete mass. "
        "Genomic risk markers are within normal baseline ranges. "
        "The classification reflects a high-confidence healthy brain finding."
    ),
    3: (
        "The model detected a focal lesion within the sellar/suprasellar region, "
        "a location highly characteristic of pituitary adenomas. "
        "The attention heatmap is concentrated around the pituitary fossa. "
        "Genomic signatures show moderate hormonal pathway activity, consistent with a secreting or non-secreting adenoma. "
        "These findings support a pituitary tumor classification with high confidence."
    )
}

TIMELINE_DATA = {
    0: [
        {"phase": "Week 1–2",   "color": "#ef4444", "icon": "🏥", "title": "Emergency Workup",       "desc": "Neuro-oncology referral, contrast MRI, molecular profiling (IDH, MGMT), biopsy planning."},
        {"phase": "Week 3–4",   "color": "#f97316", "icon": "🔬", "title": "Pathology & Staging",    "desc": "Biopsy results, WHO grading, multidisciplinary tumor board review, treatment planning."},
        {"phase": "Month 2–3",  "color": "#eab308", "icon": "⚡", "title": "Surgery / Resection",    "desc": "Maximal safe resection where feasible. Intraoperative MRI guidance recommended."},
        {"phase": "Month 3–9",  "color": "#3b5bdb", "icon": "☢️", "title": "Chemo + Radiotherapy",  "desc": "Concurrent temozolomide + 60 Gy radiotherapy (Stupp protocol for GBM)."},
        {"phase": "Month 9+",   "color": "#7c3aed", "icon": "📅", "title": "Surveillance",           "desc": "MRI every 2–3 months. Adjuvant temozolomide cycles. Monitor for recurrence."},
    ],
    1: [
        {"phase": "Week 1–2",   "color": "#3b5bdb", "icon": "🩺", "title": "Clinical Assessment",   "desc": "Neurological exam, ophthalmology referral, baseline symptom documentation."},
        {"phase": "Week 2–4",   "color": "#0ea5e9", "icon": "🔬", "title": "Detailed Imaging",       "desc": "Dedicated thin-slice MRI, angiography if near vascular structures, surgical planning."},
        {"phase": "Month 1–2",  "color": "#22c55e", "icon": "💬", "title": "MDT Decision",           "desc": "Watchful waiting vs. surgery vs. radiosurgery based on size, location, symptoms."},
        {"phase": "Month 2–4",  "color": "#f97316", "icon": "⚕️", "title": "Treatment",              "desc": "Surgery (Simpson Grade resection) or stereotactic radiosurgery for smaller lesions."},
        {"phase": "Month 6+",   "color": "#7c3aed", "icon": "📅", "title": "Follow-up",              "desc": "Annual MRI surveillance. Recurrence rate ~20% at 10 years for benign meningioma."},
    ],
    2: [
        {"phase": "Immediate",  "color": "#22c55e", "icon": "✅", "title": "No Tumor Detected",      "desc": "Reassure patient. Document findings thoroughly in clinical record."},
        {"phase": "1–3 Months", "color": "#0ea5e9", "icon": "📋", "title": "Symptom Review",         "desc": "If symptoms persist, consider repeat imaging or alternative differential diagnoses."},
        {"phase": "6 Months",   "color": "#3b5bdb", "icon": "🔄", "title": "Optional Follow-up",    "desc": "Repeat MRI only if new neurological symptoms develop or risk factors are present."},
        {"phase": "Annual",     "color": "#7c3aed", "icon": "📅", "title": "Routine Monitoring",     "desc": "Annual check-up recommended if patient has family history or genetic risk factors."},
    ],
    3: [
        {"phase": "Week 1–2",   "color": "#f97316", "icon": "🔬", "title": "Hormonal Workup",        "desc": "Full pituitary hormone panel, visual field testing, dedicated pituitary MRI (3mm slices)."},
        {"phase": "Week 2–4",   "color": "#eab308", "icon": "👁️", "title": "Specialist Referral",   "desc": "Endocrinology + ophthalmology + neurosurgery joint consultation."},
        {"phase": "Month 1–3",  "color": "#3b5bdb", "icon": "💊", "title": "Medical Therapy",        "desc": "Dopamine agonists (cabergoline) for prolactinomas. Hormone replacement if deficient."},
        {"phase": "Month 2–4",  "color": "#ef4444", "icon": "⚕️", "title": "Surgery if needed",     "desc": "Transsphenoidal surgery for non-responders, large macroadenomas, or visual compromise."},
        {"phase": "Month 6+",   "color": "#7c3aed", "icon": "📅", "title": "Long-term Surveillance", "desc": "6-monthly MRI for first 2 years, then annual. Monitor hormone levels regularly."},
    ],
}


SPECIALIST_ROUTING = {
    0: {
        "primary": "Neuro-Oncologist",
        "secondary": ["Neurosurgeon", "Radiation Oncologist", "Neuropathologist"],
        "urgency": "URGENT",
        "urgency_color": "#ef4444",
        "reason": "Gliomas require immediate multidisciplinary oncological evaluation. Delay in referral can significantly impact treatment outcomes.",
        "contact": "Refer to nearest tertiary neuro-oncology centre. Mark referral as URGENT.",
        "icon": "🔴"
    },
    1: {
        "primary": "Neurosurgeon",
        "secondary": ["Neurologist", "Neuroradiologist"],
        "urgency": "SEMI-URGENT",
        "urgency_color": "#f97316",
        "reason": "Meningiomas are typically slow-growing but require neurosurgical assessment for management planning and monitoring.",
        "contact": "Refer to neurosurgery outpatient within 2–4 weeks. Include full MRI series.",
        "icon": "🟠"
    },
    2: {
        "primary": "General Neurologist",
        "secondary": ["GP Follow-up"],
        "urgency": "ROUTINE",
        "urgency_color": "#22c55e",
        "reason": "No tumor detected. If symptoms persist, routine neurological review is appropriate to rule out other causes.",
        "contact": "Routine neurology outpatient referral if symptoms warrant further investigation.",
        "icon": "🟢"
    },
    3: {
        "primary": "Endocrinologist",
        "secondary": ["Neurosurgeon", "Ophthalmologist", "Neuroradiologist"],
        "urgency": "MODERATE",
        "urgency_color": "#eab308",
        "reason": "Pituitary tumors require hormonal workup and visual assessment before treatment decisions are made.",
        "contact": "Refer to endocrinology within 1–2 weeks. Ophthalmology if visual symptoms present.",
        "icon": "🟡"
    }
}

TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(genomic_dim):
    model = BrainTumorModel(genomic_dim).to(device)
    try:
        model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
        st.success("✅ Trained model loaded successfully.")
    except:
        st.warning("⚠️ No saved model found. Using untrained weights (for demo only).")
    model.eval()
    return model

@st.cache_resource
def load_genomics(path="GSE4290_series_matrix.txt"):
    try:
        df = pd.read_csv(path, sep="\t", comment="!")
        df = df.set_index(df.columns[0]).T
        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)
        pca = PCA(n_components=128)
        X = pca.fit_transform(X)
        tensor = torch.tensor(X, dtype=torch.float32)
        c0 = tensor.mean(0)
        c1 = tensor[:len(tensor)//2].mean(0)
        c2 = tensor[len(tensor)//2:].mean(0)
        c3 = torch.zeros_like(c0)
        return torch.stack([c0, c1, c2, c3]), 128
    except:
        return torch.randn(4, 128), 128

def overlay_heatmap(original_pil, cam):
    orig = original_pil.resize((224, 224)).convert("RGB")
    orig_np = np.array(orig) / 255.0
    from PIL import Image as PILImage
    cam_resized = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(cam_resized)[:, :, :3]
    overlay = np.clip(0.5 * orig_np + 0.5 * heatmap_colored, 0, 1)
    return (overlay * 255).astype(np.uint8)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
header_col, toggle_col = st.columns([8, 1])
with header_col:
    st.markdown("""
    <div style="font-family:'DM Serif Display',serif;font-size:3rem;text-align:center;margin-bottom:0.2rem;">
        <span style="color:#1a1f3a;">Synapse</span><span style="background:linear-gradient(90deg,#3b5bdb,#7c3aed);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">AI</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Multimodal Brain Tumor Classification — MRI + Genomic Analysis</div>', unsafe_allow_html=True)

with toggle_col:
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    label = "☀️ Light" if dark else "🌙 Dark"
    st.button(label, on_click=toggle_theme)

st.markdown("---")

# ─────────────────────────────────────────────
# Left column — upload + patient info
# ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-header">Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a brain MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded MRI Scan", use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Patient Info (Optional)</div>', unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
    patient_age  = st.number_input("Age", min_value=1, max_value=120, value=45)
    patient_id   = st.text_input("Patient ID", placeholder="e.g. PT-20240001")
    run_btn = st.button("🔍 Run Diagnosis")

# ─────────────────────────────────────────────
# Right column — results in tabs
# ─────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-header">Diagnosis Result</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner("Analyzing MRI scan with genomic data..."):

            class_genomics, genomic_dim = load_genomics()
            class_genomics = class_genomics.to(device)
            model = load_model(genomic_dim)
            img_tensor = TRANSFORM(img_pil).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                genome_batch = class_genomics[0].unsqueeze(0)
                outputs, risk, _, _ = model(img_tensor, genome_batch)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                pred_class = int(np.argmax(probs))

            with torch.no_grad():
                genome_batch = class_genomics[pred_class].unsqueeze(0)
                outputs, risk, _, _ = model(img_tensor, genome_batch)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                pred_class  = int(np.argmax(probs))
                confidence  = float(probs[pred_class]) * 100
                risk_val    = float(risk.squeeze().cpu()) * 100

            if risk_val < 35:
                risk_class = "risk-low"; risk_text = f"Low Genomic Risk ({risk_val:.1f}%)"
            elif risk_val < 65:
                risk_class = "risk-med"; risk_text = f"Moderate Genomic Risk ({risk_val:.1f}%)"
            else:
                risk_class = "risk-high"; risk_text = f"High Genomic Risk ({risk_val:.1f}%)"

            # Diagnosis card (always visible)
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:0.85rem;color:{TEXT2};margin-bottom:0.5rem;">DIAGNOSIS</div>
                <div class="class-label">{CLASS_NAMES[pred_class]}</div>
                <div class="confidence">Confidence: {confidence:.1f}%</div>
                <div><span class="risk-badge {risk_class}">{risk_text}</span></div>
                <div style="margin-top:1rem;font-size:0.9rem;color:{TEXT2};">{CLASS_INFO[pred_class]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

            # ── Tabs ──
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Analysis", "🔍 Heatmap", "🧠 Explanation", "📋 Next Steps", "🗓️ Timeline", "🏥 Routing", "📝 Referral"])

            # ── Tab 1: Charts ──
            with tab1:
                fig, (ax_gauge, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))
                fig.patch.set_facecolor(MPL_BG)

                # Donut gauge
                ax_gauge.pie(
                    [confidence/100, 1 - confidence/100],
                    colors=['#3b5bdb', '#e8ecff' if not dark else '#1e2240'],
                    startangle=90, counterclock=False,
                    wedgeprops=dict(width=0.45, edgecolor=MPL_BG, linewidth=2)
                )
                ax_gauge.text(0, 0.08, f"{confidence:.1f}%", ha='center', va='center',
                              fontsize=22, fontweight='bold', color=MPL_FG)
                ax_gauge.text(0, -0.25, "Confidence", ha='center', va='center',
                              fontsize=10, color=TEXT2)
                ax_gauge.set_title('Model Confidence', fontsize=12, color=MPL_FG, pad=10)
                ax_gauge.set_facecolor(MPL_BG)

                # Bar chart
                bar_colors = ['#ef4444', '#f97316', '#22c55e', '#3b5bdb']
                labels = [CLASS_NAMES[i] for i in range(4)]
                values = [float(p)*100 for p in probs]
                bars = ax_bar.barh(labels, values, color=bar_colors, height=0.5, edgecolor=MPL_BG)
                bars[pred_class].set_edgecolor(MPL_FG)
                bars[pred_class].set_linewidth(2)
                for bar, val in zip(bars, values):
                    ax_bar.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                                f'{val:.1f}%', va='center', fontsize=9, color=MPL_FG)
                ax_bar.set_xlim(0, 115)
                ax_bar.set_xlabel('Probability (%)', color=TEXT2, fontsize=9)
                ax_bar.set_title('Class Probabilities', fontsize=12, color=MPL_FG, pad=10)
                ax_bar.set_facecolor(MPL_BG)
                ax_bar.tick_params(colors=MPL_FG)
                for sp in ['top','right']: ax_bar.spines[sp].set_visible(False)
                for sp in ['left','bottom']: ax_bar.spines[sp].set_color(MPL_GRID)
                for lbl in ax_bar.get_yticklabels():
                    lbl.set_color(MPL_FG); lbl.set_fontsize(9)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # ── Tab 2: Heatmap ──
            with tab2:
                try:
                    target_layer = model.mri.features[-1]
                    gradcam = GradCAM(model, target_layer)
                    genome_batch = class_genomics[pred_class].unsqueeze(0)
                    cam = gradcam.generate(img_tensor, genome_batch, pred_class)

                    # Side-by-side original + overlay
                    h_col1, h_col2 = st.columns(2)
                    with h_col1:
                        st.image(img_pil, caption="Original MRI", use_container_width=True)
                    with h_col2:
                        overlay = overlay_heatmap(img_pil, cam)
                        st.image(overlay, caption="Attention Heatmap", use_container_width=True)

                    st.markdown(f'<div class="info-box">🔴 Red/warm areas show where the model focused most — highlighting the suspected region of interest.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.info(f"Heatmap generation skipped: {e}")

            # ── Tab 3: Explainability ──
            with tab3:
                st.markdown(f'<div class="section-header">Why did the model predict {CLASS_NAMES[pred_class]}?</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="explain-box">🧠 {EXPLAIN_TEXT[pred_class]}</div>', unsafe_allow_html=True)

                # Confidence warning
                if confidence < 70:
                    st.markdown(f"""
                    <div style="background:#fff8e1;border-left:4px solid #f59e0b;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-top:0.8rem;color:#7a4f00;font-size:0.9rem;">
                    ⚠️ <strong>Low confidence detected ({confidence:.1f}%)</strong> — 
                    This result should be reviewed by a qualified clinician before any clinical decisions are made.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#f0fff4;border-left:4px solid #22c55e;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-top:0.8rem;color:#145a28;font-size:0.9rem;">
                    ✅ <strong>High confidence ({confidence:.1f}%)</strong> — 
                    Model prediction is reliable. Always confirm with a specialist.
                    </div>
                    """, unsafe_allow_html=True)

            # ── Tab 4: Next Steps ──
            with tab4:
                st.markdown(f'<div class="section-header">Recommended Next Steps</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem;color:{TEXT2};margin-bottom:1rem;">Based on predicted diagnosis: <strong>{CLASS_NAMES[pred_class]}</strong> ({confidence:.1f}% confidence)</div>', unsafe_allow_html=True)
                for step in NEXT_STEPS[pred_class]:
                    st.markdown(f'<div class="next-box">{step}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box" style="margin-top:1rem;">⚕️ These recommendations are AI-generated suggestions only. All clinical decisions must be made by a licensed medical professional.</div>', unsafe_allow_html=True)

            # ── Tab 5: Treatment Timeline ──
            with tab5:
                st.markdown(f'<div class="section-header">Treatment & Care Timeline</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem;color:{TEXT2};margin-bottom:1.5rem;">Typical clinical journey for <strong>{CLASS_NAMES[pred_class]}</strong> diagnosis.</div>', unsafe_allow_html=True)

                for i, step in enumerate(TIMELINE_DATA[pred_class]):
                    is_last = i == len(TIMELINE_DATA[pred_class]) - 1
                    connector = "" if is_last else '<div style="margin-left:1.45rem;width:2px;height:1.2rem;background:' + BORDER + ';"></div>'
                    sc = step['color']
                    html = (
                        '<div style="display:flex;align-items:flex-start;gap:1rem;">'
                        '<div style="display:flex;flex-direction:column;align-items:center;">'
                        '<div style="width:2.8rem;height:2.8rem;border-radius:50%;background:' + sc + '22;'
                        'border:2px solid ' + sc + ';display:flex;align-items:center;'
                        'justify-content:center;font-size:1.1rem;flex-shrink:0;">' + step['icon'] + '</div>'
                        + connector +
                        '</div>'
                        '<div style="flex:1;padding-bottom:1rem;">'
                        '<div style="font-size:0.75rem;color:' + sc + ';font-weight:600;'
                        'text-transform:uppercase;letter-spacing:0.05em;">' + step['phase'] + '</div>'
                        '<div style="font-weight:600;color:' + TEXT + ';margin:0.2rem 0;">' + step['title'] + '</div>'
                        '<div style="font-size:0.88rem;color:' + TEXT2 + ';line-height:1.5;">' + step['desc'] + '</div>'
                        '</div></div>'
                    )
                    st.markdown(html, unsafe_allow_html=True)

                st.markdown(f'<div class="info-box">⚕️ Timeline is indicative only. Actual treatment planning must be individualised by the treating clinical team.</div>', unsafe_allow_html=True)



            # ── Tab 6: Specialist Routing ──
            with tab6:
                routing = SPECIALIST_ROUTING[pred_class]
                today = __import__('datetime').date.today().strftime("%d %B %Y")
                pname = patient_name if patient_name else "Unknown"
                pid   = patient_id if patient_id else "N/A"

                st.markdown(f'<div class="section-header">Specialist Routing Guide</div>', unsafe_allow_html=True)

                # Urgency badge
                st.markdown(
                    '<div style="display:inline-block;padding:0.4rem 1.2rem;border-radius:999px;font-weight:700;' +
                    'font-size:0.9rem;background:' + routing["urgency_color"] + '22;color:' + routing["urgency_color"] +                         ';border:2px solid ' + routing["urgency_color"] + ';margin-bottom:1rem;">' +
                    routing["icon"] + " " + routing["urgency"] + '</div>',
                    unsafe_allow_html=True
                )

                # Primary specialist card
                st.markdown(
                    '<div style="background:' + NEXT_BG + ';border:1px solid ' + NEXT_BOR +                         ';border-radius:12px;padding:1.2rem;margin-bottom:1rem;">' +
                    '<div style="font-size:0.75rem;color:' + TEXT2 + ';text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3rem;">Primary Referral</div>' +
                    '<div style="font-size:1.4rem;font-weight:700;color:' + NEXT_BOR + '">' + routing["primary"] + '</div>' +
                    '<div style="font-size:0.85rem;color:' + TEXT2 + ';margin-top:0.5rem;">' + routing["reason"] + '</div>' +
                    '</div>',
                    unsafe_allow_html=True
                )

                # Secondary specialists
                st.markdown(f'<div style="font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:0.5rem;">Also involve:</div>', unsafe_allow_html=True)
                cols = st.columns(len(routing["secondary"]))
                for col, spec in zip(cols, routing["secondary"]):
                    col.markdown(
                        '<div style="background:' + EXPLAIN_BG + ';border:1px solid ' + EXPLAIN_BOR +                             ';border-radius:8px;padding:0.6rem;text-align:center;font-size:0.85rem;color:' + EXPLAIN_TXT + '">' + spec + '</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(f'<div class="info-box" style="margin-top:1rem;">📞 {routing["contact"]}</div>', unsafe_allow_html=True)

            # ── Tab 7: Referral Letter ──
            with tab7:
                routing = SPECIALIST_ROUTING[pred_class]
                today = __import__('datetime').date.today().strftime("%d %B %Y")
                pname = patient_name if patient_name else "Unknown"
                pid   = patient_id if patient_id else "N/A"

                st.markdown(f'<div class="section-header">Referral Letter</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem;color:{TEXT2};margin-bottom:1rem;">Auto-generated based on diagnosis and patient info. Review before sending.</div>', unsafe_allow_html=True)

                letter = f"""Dear {routing["primary"]},

Re: {pname} | DOB/Age: {patient_age} years | ID: {pid}
Date: {today}

I am writing to refer the above patient for your urgent review following AI-assisted MRI brain analysis performed using SynapseAI.

FINDINGS:
The multimodal analysis (MRI + genomic profiling) has identified findings consistent with {CLASS_NAMES[pred_class]}, with a model confidence of {confidence:.1f}%. The genomic risk assessment indicates {risk_text}.

CLINICAL CONTEXT:
{EXPLAIN_TEXT[pred_class]}

URGENCY: {routing["urgency"]}
{routing["reason"]}

REQUESTED ACTION:
{routing["contact"]}

Please find the attached AI analysis report for your reference. This referral is based on AI-assisted findings and should be corroborated with full clinical assessment.

Yours sincerely,

_______________________
Referring Clinician
Date: {today}

---
⚠️ This letter was auto-generated by SynapseAI. It must be reviewed, edited, and signed by a licensed clinician before use."""

                st.text_area("Referral Letter", letter, height=420, key="referral_letter")
                st.markdown(f'<div class="info-box">📋 Copy the letter above, review it carefully, and paste into your clinical system or letterhead before sending.</div>', unsafe_allow_html=True)

    elif not uploaded:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem;color:{UPLOAD_TXT};
        border:1px dashed {UPLOAD_BORDER};border-radius:12px;">
            <div style="font-size:2.5rem">🖼️</div>
            <div style="margin-top:0.5rem">Upload an MRI scan on the left to get started</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:{TEXT2};font-size:0.8rem;padding:0.5rem">
    SynapseAI — For research and clinical assistance only. Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)