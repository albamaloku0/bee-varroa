
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Konfigurimi i faqes ──────────────────────────────────
st.set_page_config(
    page_title="🐝 Bee Health Monitor",
    page_icon="🐝",
    layout="centered"
)

# ── Ngarkojmë modelin (vetëm një herë) ───────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/content/bee_model_final.keras")

model = load_model()
IMG_SIZE = (224, 224)

# ── Funksione ndihmëse ────────────────────────────────────
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[:, :, :3]
    return np.expand_dims(arr, axis=0), arr

def predict(img_array):
    proba = model.predict(img_array, verbose=0)[0][0]
    if proba > 0.5:
        return "varroa", float(proba)
    else:
        return "healthy", float(1 - proba)

# ── UI ────────────────────────────────────────────────────
st.title("🐝 Bee Health Monitor")
st.markdown("**Zbulimi automatik i parazitit Varroa destructor me Inteligjencë Artificiale**")
st.markdown("---")

# Sidebar me info
with st.sidebar:
    st.header("ℹ️ Rreth Modelit")
    st.markdown("""
    **Modeli:** MobileNetV2  
    **Trajnimi:** Transfer Learning  
    **Klasa:** Healthy / Varroa  
    **Dataset:** 4,435 imazhe  
    """)
    st.markdown("---")
    st.markdown("**Si funksionon:**")
    st.markdown("1. Ngarko një foto blete")
    st.markdown("2. AI analizon imazhin")
    st.markdown("3. Merr rezultatin menjëherë")

# Upload imazhi
uploaded = st.file_uploader(
    "📸 Ngarko një foto blete:",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Imazhi i ngarkuar")
        st.image(image, use_column_width=True)

    # Parashikimi
    with st.spinner("🔍 Duke analizuar..."):
        img_array, img_arr = preprocess_image(image)
        pred_class, confidence = predict(img_array)

    with col2:
        st.subheader("🧠 Rezultati i AI-së")

        if pred_class == "healthy":
            st.success(f"✅ HEALTHY — Bletë e Shëndetshme")
            st.markdown(f"**Besueshmëria:** {confidence*100:.1f}%")
            st.progress(confidence)
            st.markdown("🐝 Kjo bletë nuk tregon shenja të infestimit me Varroa.")
        else:
            st.error(f"⚠️ VARROA — Bletë e Infestuar")
            st.markdown(f"**Besueshmëria:** {confidence*100:.1f}%")
            st.progress(confidence)
            st.markdown("🦟 Kjo bletë tregon shenja të infestimit me Varroa destructor.")
            st.markdown("**Rekomandim:** Konsultohuni me bletarin për trajtim menjëherë.")

    # Grafiku i probabilitetit
    st.markdown("---")
    st.subheader("📊 Distribucioni i Probabilitetit")

    raw_proba = model.predict(img_array, verbose=0)[0][0]
    probs     = [1 - raw_proba, raw_proba]
    labels    = ["Healthy", "Varroa"]
    colors    = ["#4CAF50", "#FF5722"]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(labels, probs, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabiliteti")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold 50%")
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{prob*100:.1f}%", va="center", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("👆 Ngarko një foto blete për të filluar analizën.")
    st.markdown("---")
    st.markdown("### 📌 Shembuj imazhesh që mund të testosh:")
    st.markdown("Imazhet duhet të jenë me rezolucion të mjaftueshëm dhe të tregojnë qartë bletën.")
