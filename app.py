import os
import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np

from src.streamlit_model import load_model
from src.streamlit_preprocessing import preprocess_image
from src.streamlit_inference import run_inference

# ── Model paths ────────────────────────────────────────────
CHECKPOINTS = {
    "EfficientNetB0":    "models/efficientnet_best_weights.pth",
    "MobileNetV3-Small": "models/mobilenet_best_weights.pth",
}

MODEL_INFO = {
    "EfficientNetB0": {
        # Sumber: BAB4_MATERIAL_REVISION_V4 — benchmark.csv (epoch_40, 2026-08-20)
        #         & FINAL_TEST_MAE_SUMMARY.csv
        "params":     "4.011.391",
        "trainable":  "3.843",
        "flops":      "0.4139 GFLOPs",
        "latency":    "17.83 ms",
        "fps":        "56.10 FPS",
        "best_epoch": "Epoch 40 / 50",
        "val_mae":    "2.9981°",
        "mae_aflw":   "11.54°",
        "mae_biwi":   "8.27°",
        "desc":       "Akurasi lebih tinggi · 50 epoch · Best: Epoch 40",
    },
    "MobileNetV3-Small": {
        # Sumber: BAB4_MATERIAL_REVISION_V4 — benchmark.csv (epoch_49, 2026-08-20)
        #         & FINAL_TEST_MAE_SUMMARY.csv
        "params":     "1.520.931",
        "trainable":  "593.923",
        "flops":      "0.0615 GFLOPs",
        "latency":    "10.77 ms",
        "fps":        "92.86 FPS",
        "best_epoch": "Epoch 49 / 50",
        "val_mae":    "4.1297°",
        "mae_aflw":   "12.87°",
        "mae_biwi":   "10.04°",
        "desc":       "Sangat ringan & cepat · 50 epoch · Best: Epoch 49",
    },
}

def angle_to_direction(angle: float, axis: str) -> str:
    if axis == "yaw":
        if angle > 15:  return "Kanan"
        if angle < -15: return "Kiri"
        return "Lurus"
    if axis == "pitch":
        if angle > 15:  return "Atas"
        if angle < -15: return "Bawah"
        return "Lurus"
    if axis == "roll":
        if angle > 15:  return "Miring Kanan"
        if angle < -15: return "Miring Kiri"
        return "Tegak"
    return ""

def severity_label(angle: float) -> str:
    a = abs(angle)
    if a < 15:  return "Normal"
    if a < 45:  return "Sedang"
    return "Ekstrem"


def main():
    st.set_page_config(
        page_title="Head Pose Estimation",
        page_icon=":mortar_board:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        font-size: 0.9rem;
        opacity: 0.7;
        margin: 0.4rem 0 0 0;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.2rem 1rem;
        color: white;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.72rem;
        font-weight: 500;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
        margin: 0.25rem 0;
    }
    .metric-card .sub {
        font-size: 0.78rem;
        opacity: 0.7;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(102,126,234,0.12);
        color: #4a5fb5;
        border: 1px solid rgba(102,126,234,0.25);
        margin: 0.15rem 0.1rem;
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 1.4rem 0 0.6rem 0;
    }

    .result-panel {
        background: #f8f9ff;
        border: 1px solid #e2e6f5;
        border-radius: 10px;
        padding: 1.4rem;
    }

    .model-stat {
        font-size: 0.81rem;
        color: #555;
        margin: 0.22rem 0;
        line-height: 1.5;
    }

    .stat-section-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #999;
        margin: 0.7rem 0 0.3rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>Head Pose Estimation</h1>
        <p>Estimasi sudut kepala (Yaw · Pitch · Roll) berbasis Deep Learning &mdash; Tugas Akhir S1 Teknik Informatika</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='stat-section-label'>Model</div>", unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Pilih model backbone:",
            options=list(CHECKPOINTS.keys()),
            index=0,
            label_visibility="collapsed",
        )

        info = MODEL_INFO[selected_model]

        s_row  = "font-size:0.81rem;color:#444;margin:0.22rem 0;line-height:1.5;"
        s_head = "font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#999;margin:0.7rem 0 0.25rem 0;"

        card_html = (
            '<div style="background:#f4f6ff;border-radius:10px;padding:1rem;margin-top:0.4rem;border:1px solid #e2e6f5;">'
            + '<div style="font-weight:700;font-size:0.9rem;color:#1a1a2e;">' + selected_model + "</div>"
            + '<div style="font-size:0.75rem;color:#888;margin:0.2rem 0 0.8rem 0;">' + info["desc"] + "</div>"
            + '<div style="' + s_head + '">Arsitektur</div>'
            + '<div style="' + s_row + '"><b>Total Parameters</b> &nbsp; ' + info["params"] + "</div>"
            + '<div style="' + s_row + '"><b>Trainable Params</b> &nbsp; ' + info["trainable"] + "</div>"
            + '<div style="' + s_row + '"><b>GFLOPs</b> &nbsp; ' + info["flops"] + "</div>"
            + '<div style="' + s_head + '">Performa Inferensi</div>'
            + '<div style="' + s_row + '"><b>Latency</b> &nbsp; ' + info["latency"] + "</div>"
            + '<div style="' + s_row + '"><b>Throughput</b> &nbsp; ' + info["fps"] + "</div>"
            + '<div style="' + s_row + '"><b>Best Checkpoint</b> &nbsp; ' + info["best_epoch"] + "</div>"
            + '<div style="' + s_head + '">Evaluasi Eksternal</div>'
            + '<div style="' + s_row + '"><b>Val MAE (300W-LP)</b> &nbsp; ' + info["val_mae"] + "</div>"
            + '<div style="' + s_row + '"><b>MAE AFLW2000-3D</b> &nbsp; ' + info["mae_aflw"] + "</div>"
            + '<div style="' + s_row + '"><b>MAE BIWI</b> &nbsp; ' + info["mae_biwi"] + "</div>"
            + "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")

        checkpoint_path = CHECKPOINTS[selected_model]
        with st.spinner("Memuat model..."):
            try:
                model = load_model(selected_model, checkpoint_path)
                st.success("Model berhasil dimuat")
            except FileNotFoundError:
                st.error(f"Checkpoint tidak ditemukan: `{checkpoint_path}`")
                st.stop()
            except RuntimeError as e:
                st.error(f"Error memuat model: {e}")
                st.stop()

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.75rem;color:#999;line-height:1.7;'>
            <div style='font-weight:600;color:#777;margin-bottom:0.3rem;'>Tentang Aplikasi</div>
            Model dilatih pada dataset <b>300W-LP</b> dan dievaluasi pada
            <b>AFLW2000-3D</b> dan <b>BIWI</b>.<br>
            Output berupa regresi sudut kontinyu, bukan klasifikasi.<br>
            Optimizer: Adam &nbsp;&middot;&nbsp; LR: 1e-4 &nbsp;&middot;&nbsp; 50 epoch
        </div>
        """, unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Input Gambar</div>", unsafe_allow_html=True)
    tab_upload, tab_camera = st.tabs(["Upload File", "Kamera"])

    image_to_process = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Pilih file gambar (JPG / PNG):",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                image_to_process = np.array(pil_image)
            except UnidentifiedImageError:
                st.error("File tidak dapat dibaca. Harap unggah gambar yang valid.")

    with tab_camera:
        camera_file = st.camera_input("Ambil foto dari kamera:")
        if camera_file is not None:
            try:
                pil_image = Image.open(camera_file).convert("RGB")
                image_to_process = np.array(pil_image)
            except Exception as e:
                st.error(f"Gagal membaca kamera: {e}")

    # ── Inference ─────────────────────────────────────────
    if image_to_process is not None:
        st.markdown("<div class='section-title'>Hasil Inferensi</div>", unsafe_allow_html=True)

        fallback_option = st.checkbox(
            "Gunakan seluruh gambar jika wajah tidak terdeteksi (fallback mode)",
            value=True,
        )

        col_img, col_crop, col_result = st.columns([1, 1, 2], gap="large")

        with col_img:
            st.markdown("**Gambar Asli**")
            st.image(image_to_process, use_column_width=True)

        with st.spinner("Memproses gambar..."):
            tensor, face_crop, status = preprocess_image(
                image_to_process, fallback_full_image=fallback_option
            )

        if status == "no_face_failed":
            st.warning(
                "Wajah tidak terdeteksi. Aktifkan fallback mode di atas "
                "untuk memproses seluruh gambar."
            )
        else:
            with col_crop:
                st.markdown("**Crop Wajah**")
                st.image(face_crop, use_column_width=True)
                if status == "no_face_fallback":
                    st.caption("Menggunakan seluruh gambar (fallback).")

            with col_result:
                st.markdown(f"**Prediksi Sudut — {selected_model}**")

                with st.spinner("Menjalankan inferensi..."):
                    pitch, yaw, roll = run_inference(model, tensor)

                c1, c2, c3 = st.columns(3)
                angles = {"Yaw (Y)": yaw, "Pitch (X)": pitch, "Roll (Z)": roll}
                axes   = ["yaw", "pitch", "roll"]

                for col, (label, val), ax in zip([c1, c2, c3], angles.items(), axes):
                    direction = angle_to_direction(val, ax)
                    severity  = severity_label(val)
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="label">{label}</div>
                            <div class="value">{val:+.1f}&deg;</div>
                            <div class="sub">{direction} &middot; {severity}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                mean_abs = (abs(pitch) + abs(yaw) + abs(roll)) / 3
                st.markdown(f"""
                <div class="result-panel">
                    <div style='font-weight:600;font-size:0.9rem;margin-bottom:0.7rem;'>Ringkasan</div>
                    <div style='font-size:0.88rem;line-height:2;'>
                        <span class='badge'>Yaw &nbsp; {yaw:+.2f}&deg;</span>
                        <span class='badge'>Pitch &nbsp; {pitch:+.2f}&deg;</span>
                        <span class='badge'>Roll &nbsp; {roll:+.2f}&deg;</span><br>
                        <b>Rata-rata absolut:</b> &nbsp; {mean_abs:.2f}&deg;<br>
                        <b>Orientasi:</b> &nbsp;
                        {angle_to_direction(yaw,'yaw')} &middot;
                        {angle_to_direction(pitch,'pitch')} &middot;
                        {angle_to_direction(roll,'roll')}
                    </div>
                    <div style='font-size:0.73rem;color:#aaa;margin-top:0.9rem;
                                border-top:1px solid #e2e6f5;padding-top:0.7rem;line-height:1.7;'>
                        Output berupa regresi sudut kontinyu (bukan klasifikasi kategori).
                        Konvensi sudut mengikuti anotasi dataset 300W-LP / AFLW2000-3D.
                        Latensi pada cloud dapat berbeda dari benchmark lokal.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;font-size:0.75rem;color:#bbb;padding:0.3rem 0;'>
        Tugas Akhir &mdash; Estimasi Pose Kepala Berbasis Deep Learning &nbsp;&middot;&nbsp;
        EfficientNetB0 &amp; MobileNetV3-Small &nbsp;&middot;&nbsp;
        Dataset: 300W-LP &middot; AFLW2000-3D &middot; BIWI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
