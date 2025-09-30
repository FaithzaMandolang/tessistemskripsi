# app.py
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from dotenv import load_dotenv
import markdown

# Optional LLM client (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnetrms (1).h5"   # ganti sesuai nama model
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
        llm_client = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Gemini LLM configured (gemini-2.5-flash).")
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
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(keras_model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("Tidak menemukan layer convolution.")

def _ensure_tensor(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Object is empty list/tuple where tensor expected.")
        return x[0]
    return x

def make_gradcam_overlay(keras_model, img_rgb_uint8, x_preprocessed, pred_class, alpha=0.4, layer_name=None):
    if layer_name is None:
        layer_name = find_last_conv_layer(keras_model)

    try:
        layer_output = keras_model.get_layer(layer_name).output
    except Exception as e:
        raise ValueError(f"Layer {layer_name} tidak ditemukan: {e}")

    grad_model = tf.keras.models.Model(keras_model.inputs, [layer_output, keras_model.output])

    x = tf.convert_to_tensor(x_preprocessed, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        conv_outputs = _ensure_tensor(conv_outputs)
        predictions = _ensure_tensor(predictions)
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)
    grads = _ensure_tensor(grads)

    if grads is None:
        raise RuntimeError("Gradien None — tidak dapat menghitung gradien.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_arr = conv_outputs[0].numpy()
    pooled_grads_arr = pooled_grads.numpy()

    weighted = conv_outputs_arr * pooled_grads_arr[np.newaxis, np.newaxis, :]

    heatmap = np.mean(weighted, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    h, w = img_rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color_bgr, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_resized, overlay_rgb

def save_resized_for_llm(overlay_rgb_uint8, path_out: Path, max_size=OVERLAY_LLM_MAX):
    pil = Image.fromarray(overlay_rgb_uint8)
    pil.thumbnail((max_size, max_size))
    pil.save(str(path_out), format="JPEG", quality=85)

# ---------------- Prompt builders ----------------
def build_explanation_prompt(pred_label: str):
    return f"""
Kamu adalah asisten kecantikan.

Hasil prediksi model: {pred_label}.

Tugasmu:
1. Jelaskan hasil prediksi model dengan bahasa sederhana agar mudah dipahami.
2. Gunakan visualisasi Grad-CAM (overlay warna merah/oranye) untuk menjelaskan area wajah yang diperhatikan model.
3. Terangkan kenapa area tersebut relevan dengan kondisi kulit {pred_label}.
4. Gunakan bahasa ringan, ringkas, dan mudah dimengerti orang awam.
"""

def build_recommendation_prompt(pred_label: str):
    return f"""
Kamu adalah ahli skincare.

Hasil prediksi model: {pred_label}.

Tugasmu:
1. Rekomendasikan **zat aktif skincare** yang sesuai untuk kondisi kulit {pred_label}.
2. Berikan contoh **produk skincare nyata** (brand global/umum) yang mengandung zat aktif tersebut.
   - Sebutkan nama produk
   - Sebutkan zat aktif utama
   - Jelaskan singkat kenapa produk itu cocok
3. Ingatkan bahwa ini hanya saran umum berbasis AI, bukan diagnosis medis.

Format keluaran:
- Daftar poin untuk zat aktif
- Daftar poin untuk produk skincare
- Penutup berupa disclaimer singkat
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

        img_orig = Image.open(str(save_path)).convert("RGB")
        img_orig_rgb = np.array(img_orig)

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

        explanation_text = None
        recommendation_text = None
        if llm_client is not None:
            try:
                # Prompt 1: Penjelasan
                prompt1 = build_explanation_prompt(pred_label)
                img_for_llm = Image.open(str(overlay_llm_save))
                resp1 = llm_client.generate_content([prompt1, img_for_llm])
                explanation_text = getattr(resp1, "text", None) or str(resp1)

                # Prompt 2: Rekomendasi
                prompt2 = build_recommendation_prompt(pred_label)
                resp2 = llm_client.generate_content(prompt2)
                recommendation_text = getattr(resp2, "text", None) or str(resp2)

                # Format agar rapi di HTML
                import markdown
                explanation_text = markdown.markdown(explanation_text, extensions=["extra"])
                recommendation_text = markdown.markdown(recommendation_text, extensions=["extra"])

            except Exception as e:
                explanation_text = "LLM gagal menjelaskan prediksi."
                recommendation_text = "LLM gagal memberi rekomendasi."

        return render_template(
            "result.html",
            prediction=pred_label,
            confidence=round(confidence, 3),
            original=url_for("static", filename=f"uploads/{orig_save.name}"),
            heatmap=url_for("static", filename=f"uploads/{heat_save.name}"),
            overlay=url_for("static", filename=f"uploads/{overlay_save.name}"),
            explanation=explanation_text,
            recommendation=recommendation_text
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
