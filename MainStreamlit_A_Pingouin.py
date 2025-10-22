import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from pathlib import Path

from streamlit_option_menu import option_menu

# Navigasi sidebar
with st.sidebar:
    selected = option_menu("Tugas UTS Streamlit UTS ML 24/25",
                            ['Upload Dataset',
                            'Klasifikasi Cuaca',
                            'Catatan'],
                            default_index=0)
    
if selected == 'Upload Dataset':
    st.header("")


CSV_PATH = "dpc.csv"
MODEL_PATH = "BestModel_CLF_RandomForest_pingouin.pkl"
UI_FEATURES = ["Suhu (°C)", "Kelembapan (%)"]
LABEL_MAP = {0: "Hujan", 1: "Cerah"}

st.set_page_config(page_title="Prediksi Cuaca", page_icon="🌤️", layout="centered")
st.title("🌦️ Prediksi Cuaca (Klasifikasi)") 

st.markdown("""
<style>
/* Warna dasar aplikasi */
.stApp {
    background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #f0f0f0;
}

/* Panel transparan di atas background hitam */
.panel {
    padding: 18px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* Tombol oranye gradasi */
.stButton > button {
    background: linear-gradient(90deg,#ff9966 0%, #ff5e62 100%);
    border: none;
    color: white;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight:700;
    box-shadow: 0 10px 20px rgba(255,94,98,.35);
}
.stButton > button:hover {
    filter: brightness(1.1);
}

/* Teks & Header */
.title {
    font-weight: 800;
    font-size: 32px;
    background: linear-gradient(90deg,#36d1dc 0%,#5b86e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.subtle {
    color: #ccc;
    margin-bottom: 8px;
}

/* Nama tim animasi warna */
.rainbow {
    display:inline-block;
    font-weight:800;
    padding:4px 10px;
    border-radius:10px;
    background:linear-gradient(90deg,#ff6a88,#ffcc70,#a1c4fd,#c2ffd8,#ff6a88);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:hueCycle 6s linear infinite;
}
@keyframes hueCycle {0%{filter:hue-rotate(0)}100%{filter:hue-rotate(360deg)}}

/* Chip probabilitas */
.metric-chip {
    display:inline-block;
    padding:10px 16px;
    border-radius:999px;
    background:linear-gradient(90deg,#36d1dc 0%,#5b86e5 100%);
    color:#fff;
    font-weight:700;
    box-shadow:0 8px 20px rgba(91,134,229,.28);
    margin-top:8px;
}

/* Footer */
.footer {
    text-align:center;
    color:#aaa;
    font-size:13px;
    margin-top:12px;
}
</style>

<div class="wrap">
    <div class="subtle">
        Masukkan <b>Suhu (°C)</b> & <b>Kelembapan (%)</b>. Fitur lain dilengkapi otomatis dari median dataset.
    </div>
    <div class="subtle">Dibuat oleh <span class="rainbow">BENY, DENIS, RENALDI</span></div>
</div>
""", unsafe_allow_html=True)

if not Path(CSV_PATH).exists():
    st.error(f"File {CSV_PATH} tidak ditemukan di folder ini.")
    st.stop()
if not Path(MODEL_PATH).exists():
    st.error(f"File model {MODEL_PATH} tidak ditemukan di folder ini.")
    st.stop()

df = pd.read_csv(CSV_PATH)
model = joblib.load(MODEL_PATH)

st.write('Untuk Inputan File dataset (csv) bisa menggunakan st.file_uploader')
file = st.file_uploader("Masukkan File", type=["csv", "txt"])
if file is not None:
    try:
        df = pd.read_csv(file)
        st.success("File berhasil diunggah!")
        st.dataframe(df)  # menampilkan isi CSV di Streamlit
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")

def ensure_features(X_input, model, df):
    expected = list(model.feature_names_in_)
    for f in expected:
        if f not in X_input.columns:
            if f in df.columns:
                X_input[f] = pd.to_numeric(df[f], errors="coerce").median()
            else:
                X_input[f] = 0
    return X_input[expected].apply(pd.to_numeric, errors="coerce")

def pretty_label(y_pred, model):
    if hasattr(model, "classes_"):
        if isinstance(y_pred, str):
            return y_pred
        return LABEL_MAP.get(int(y_pred), str(y_pred))
    return LABEL_MAP.get(int(y_pred), str(y_pred))


st.markdown('<div class="wrap"><div class="panel">', unsafe_allow_html=True)


if selected == 'Klasifikasi Cuaca':
    st.header("Input Fitur Cuaca")
c1, c2 = st.columns(2)
with c1:
    suhu = st.number_input("Suhu (°C)", value=float(pd.to_numeric(df["Suhu (°C)"], errors="coerce").median()))
with c2:
    kelembapan = st.number_input("Kelembapan (%)", value=float(pd.to_numeric(df["Kelembapan (%)"], errors="coerce").median()))
st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('<div class="wrap"><div class="panel">', unsafe_allow_html=True)
X_user = pd.DataFrame([[suhu, kelembapan]], columns=UI_FEATURES)
X = ensure_features(X_user, model, df)
if st.button("🔍 Prediksi Cuaca"):
    try:
        y_pred = model.predict(X)[0]
        hasil = pretty_label(y_pred, model)
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        st.success(f"Hasil Prediksi: **{hasil}**")
        if proba is not None and hasattr(model, "classes_"):
            kelas = [pretty_label(c, model) for c in model.classes_]
            prob_df = pd.DataFrame({"Kelas": kelas, "Probabilitas": proba})
            color_range = ["#7bdff2", "#f2b5d4", "#b2f7ef", "#f7d6e0", "#c5d6ff"][:len(prob_df)]
            chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
                x=alt.X("Kelas:N", title="Kelas Cuaca"),
                y=alt.Y("Probabilitas:Q", title="Probabilitas", scale=alt.Scale(domain=[0,1])),
                color=alt.Color("Kelas:N", scale=alt.Scale(range=color_range), legend=None),
                tooltip=[alt.Tooltip("Kelas:N"), alt.Tooltip("Probabilitas:Q", format=".2%")]
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
            st.markdown(f'<div class="metric-chip"> Probabilitas Prediksi: {proba[np.argmax(proba)]*100:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.info("Model tidak menyediakan probabilitas (predict_proba).")
    except Exception as e:
        st.error(f"Gagal memprediksi: {e}")
st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('<div class="wrap"><div class="footer">Made with • Beny Denis Renaldi</div></div>', unsafe_allow_html=True)


if selected == 'Catatan':
    st.header("📝 Catatan Penting – Aplikasi Prediksi Cuaca")

    st.markdown("""
    ### 🧠 **1. Tujuan Aplikasi**
    Aplikasi ini digunakan untuk **memprediksi kondisi cuaca (Hujan atau Cerah)** berdasarkan **parameter suhu (°C)** dan **kelembapan (%)** menggunakan model klasifikasi **Random Forest** yang telah dilatih sebelumnya.  
    Tujuannya adalah memberikan simulasi **penerapan Machine Learning (ML)** di bidang **analisis data cuaca** dalam bentuk **web interaktif berbasis Streamlit**.
    ### 📈 **2. Fitur Input & Output**
    **Input:**
    - Suhu (°C)
    - Kelembapan (%)
    - Dataset eksternal (opsional)

    **Output:**
    - Prediksi kondisi cuaca: 🌧️ *Hujan* atau ☀️ *Cerah*  
    - Probabilitas hasil prediksi dalam bentuk grafik (Altair).  
    - Nilai probabilitas tertinggi ditampilkan dalam *metric chip* berwarna.

    ---
    """)