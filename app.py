# app.py
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from dotenv import load_dotenv

# Optional LLM client (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnetrms (1).h5"   # ganti nama sesuai model
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = (224, 224)
OVERLAY_LLM_MAX = 512

LABEL_LIST_PATH = BASE_DIR / "label_list.npy"
if LABEL_LIST_PATH.exists():
    LABEL_LIST = list(np.load(str(LABEL_LIST_PATH), allow_pickle=True))
else:
    LABEL_LIST = ["Berminyak", "Dark Spots", "Jerawat", "Kemerahan", "Kering", "Kerutan"]

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load .env & configure LLM ----------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
llm_client = None
if genai is not None and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        llm_client = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini LLM configured (gemini-1.5-flash).")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini: {e}")
        llm_client = None
else:
    logger.info("Gemini not configured.")

# ---------------- Load Keras model ----------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model .h5 tidak ditemukan di: {MODEL_PATH}")
logger.info("Loading Keras model...")
model = load_model(str(MODEL_PATH))
logger.info("Model loaded.")

# ---------------- Helper functions ----------------
def allowed_file(filename: str):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def find_last_conv_layer(keras_model):
    # cari Conv2D terakhir, fallback cari nama yg mengandung 'conv'
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(keras_model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("Tidak menemukan layer convolution.")

def _ensure_tensor(x):
    """Helper: bila x adalah list/tuple ambil elemen pertama; kembalikan tf.Tensor."""
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Object is empty list/tuple where tensor expected.")
        return x[0]
    return x

def make_gradcam_overlay(keras_model, img_rgb_uint8, x_preprocessed, pred_class, alpha=0.4, layer_name=None):
    """
    Robust Grad-CAM:
      - menangani kasus layer.output/grad yang returned as list/tuple
      - mengembalikan heatmap_resized (float 0..1) dan overlay_rgb (uint8)
    Args:
      keras_model: model
      img_rgb_uint8: original image RGB uint8 (H,W,3)
      x_preprocessed: input preprocessed for model (1,H,W,3) - tf tensor or np array
      pred_class: int index
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(keras_model)

    # prepare model that outputs conv maps + predictions
    try:
        layer_output = keras_model.get_layer(layer_name).output
    except Exception as e:
        raise ValueError(f"Layer {layer_name} tidak ditemukan: {e}")

    grad_model = tf.keras.models.Model(keras_model.inputs, [layer_output, keras_model.output])

    # convert x_preprocessed to tensor float32
    x = tf.convert_to_tensor(x_preprocessed)
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)

    with tf.GradientTape() as tape:
        # forward pass
        conv_outputs, predictions = grad_model(x)
        # ensure conv_outputs is a tensor (not list)
        conv_outputs = _ensure_tensor(conv_outputs)
        predictions = _ensure_tensor(predictions)
        # loss: score of the target class
        loss = predictions[:, pred_class]

    # compute gradients of the class output w.r.t conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    grads = _ensure_tensor(grads)

    if grads is None:
        raise RuntimeError("Gradien None — tidak dapat menghitung gradien (cek model/input).")

    # pooled grads: average over spatial dims (H, W) and batch (0)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # convert to numpy arrays for channel weighting
    conv_outputs_arr = conv_outputs[0].numpy()  # shape: (H, W, C)
    pooled_grads_arr = pooled_grads.numpy()     # shape: (C,)

    # safety checks shapes
    if conv_outputs_arr.ndim != 3:
        raise RuntimeError(f"conv_outputs_arr harus 3D (H,W,C), tapi bentuknya: {conv_outputs_arr.shape}")
    if pooled_grads_arr.ndim != 1:
        raise RuntimeError(f"pooled_grads_arr harus 1D (C,), tapi bentuknya: {pooled_grads_arr.shape}")

    # weight channels
    # broadcasting multiply: safer and faster than loop
    try:
        weighted = conv_outputs_arr * pooled_grads_arr[np.newaxis, np.newaxis, :]
    except Exception:
        # fallback ke loop jika broadcasting gagal
        for i in range(pooled_grads_arr.shape[-1]):
            conv_outputs_arr[:, :, i] *= pooled_grads_arr[i]
        weighted = conv_outputs_arr

    heatmap = np.mean(weighted, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # resize heatmap ke resolusi gambar asli
    h, w = img_rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # overlay: convert original RGB -> BGR, addWeighted, back to RGB
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color_bgr, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_resized, overlay_rgb

def save_resized_for_llm(overlay_rgb_uint8, path_out: Path, max_size=OVERLAY_LLM_MAX):
    pil = Image.fromarray(overlay_rgb_uint8)
    pil.thumbnail((max_size, max_size))
    pil.save(str(path_out), format="JPEG", quality=85)

def build_llm_prompt(pred_label: str):
    return f"""
Kamu adalah asisten kecantikan yang menjelaskan hasil klasifikasi kondisi kulit wajah
dengan cara sederhana agar mudah dipahami orang awam.

Input:
- Hasil prediksi model: {pred_label}
- Visualisasi Grad-CAM overlay yang menunjukkan area wajah yang paling diperhatikan model.

Tugasmu:
1. Jelaskan hasil prediksi model dengan bahasa awam, tanpa istilah medis yang rumit.
2. Terangkan area wajah yang ditandai warna merah/oranye pada Grad-CAM overlay
   sebagai area yang diperhatikan model.
3. Berikan penjelasan sederhana kenapa area itu penting.
4. Rekomendasikan **zat aktif skincare** yang sesuai untuk mengatasi masalah kulit hasil prediksi.
5. Berikan contoh **produk skincare nyata** (brand global/umum) yang mengandung zat aktif tersebut.
   - Sebutkan nama produk
   - Sebutkan zat aktif utama di dalam produk
   - Jelaskan singkat kenapa produk itu cocok
6. Ingatkan bahwa ini hanyalah saran umum berbasis AI, bukan diagnosis medis atau rekomendasi dokter.

Format keluaran:
- Paragraf singkat (penjelasan hasil & Grad-CAM).
- Rekomendasi zat aktif (dalam bentuk daftar poin).
- Rekomendasi produk skincare (dalam bentuk daftar poin).
- Penutup berupa disclaimer singkat.
"""

# ---------------- Flask app ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="Tidak menemukan file upload.")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="Nama file kosong.")
        if not allowed_file(file.filename):
            return render_template("index.html", error="File harus jpg/jpeg/png.")

        filename = secure_filename(file.filename)
        base, _ = os.path.splitext(filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(str(save_path))

        # load gambar asli untuk display (ORIGINAL, full resolution)
        img_orig = Image.open(str(save_path)).convert("RGB")
        img_orig_rgb = np.array(img_orig)

        # resize untuk model (keep aspect ratio bias di sini kami gunakan simple resize)
        pil_for_model = img_orig.resize(TARGET_SIZE)
        img_array = keras_image.img_to_array(pil_for_model)
        x = np.expand_dims(img_array.copy(), axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        pred_class = int(np.argmax(preds, axis=1)[0])
        pred_label = LABEL_LIST[pred_class] if pred_class < len(LABEL_LIST) else f"Class-{pred_class}"
        confidence = float(np.max(preds))

        try:
            heatmap_resized, overlay_rgb = make_gradcam_overlay(model, img_orig_rgb, x, pred_class)
        except Exception as e:
            logger.exception("Grad-CAM gagal")
            return render_template("index.html", error=f"Grad-CAM error: {e}")

        # simpan file untuk web
        orig_save = UPLOAD_FOLDER / f"{base}_orig.jpg"
        heat_save = UPLOAD_FOLDER / f"{base}_heat.jpg"
        overlay_save = UPLOAD_FOLDER / f"{base}_overlay.jpg"
        overlay_llm_save = UPLOAD_FOLDER / f"{base}_overlay_llm.jpg"

        img_orig.save(str(orig_save))
        heat_uint8 = np.uint8(255 * heatmap_resized)
        heat_col_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_col_rgb = cv2.cvtColor(heat_col_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat_col_rgb).save(str(heat_save))
        Image.fromarray(overlay_rgb).save(str(overlay_save))
        save_resized_for_llm(overlay_rgb, overlay_llm_save)

        llm_text = None
        if llm_client is not None:
            try:
                prompt = build_llm_prompt(pred_label)
                img_for_llm = Image.open(str(overlay_llm_save))
                resp = llm_client.generate_content([prompt, img_for_llm])
                llm_text = getattr(resp, "text", None) or str(resp)
            except Exception:
                llm_text = "LLM gagal merespon."

        return render_template(
            "result.html",
            prediction=pred_label,
            confidence=round(confidence, 3),
            original=url_for("static", filename=f"uploads/{orig_save.name}"),
            heatmap=url_for("static", filename=f"uploads/{heat_save.name}"),
            overlay=url_for("static", filename=f"uploads/{overlay_save.name}"),
            explanation=llm_text
        )

    return render_template("index.html")

@app.route("/test-gemini")
def test_gemini():
    if llm_client is None:
        return "Gemini not configured."
    try:
        resp = llm_client.generate_content("Halo, ini tes koneksi ke Gemini.")
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"Gemini error: {e}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
