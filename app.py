from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import io
import uuid
import json
from flask import send_file

MODEL_PATH = os.path.join("evidencia_entrenamiento", "inceptionv3", "inceptionv3_final.keras")

app = Flask(__name__, static_folder="static")
CORS(app)

print("Cargando modelo desde:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Determinar tamaño de entrada esperado por el modelo
input_shape = model.input_shape
if input_shape and len(input_shape) >= 3:
    target_size = (int(input_shape[1]), int(input_shape[2]))
else:
    target_size = (224, 224)

LABELS_FILE = os.path.join(os.path.dirname(__file__), 'banana_class_indices.json')
CLASSES = ["Premium", "Rechazo"]
try:
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        # mapping expected as {label: index}
        inv = {int(v): k for k, v in mapping.items()}
        max_idx = max(inv.keys())
        classes_from_file = [inv.get(i, f"Class_{i}") for i in range(max_idx + 1)]
        CLASSES = classes_from_file
        print("Cargando etiquetas desde", LABELS_FILE, "->", CLASSES)
except Exception as e:
    print("No se pudo leer", LABELS_FILE, e)


def prepare_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        # layer.output_shape can be a tuple like (None, h, w, c)
        try:
            shape = layer.output_shape
        except Exception:
            continue
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer.name
    return None


def make_gradcam(img_array, model, class_index):
    last_conv_name = find_last_conv_layer(model)
    if last_conv_name is None:
        raise RuntimeError('No se encontró capa convolucional en el modelo')

    last_conv = model.get_layer(last_conv_name)

    # Create a model that maps the input image to the activations
    grad_model = tf.keras.models.Model([
        model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads

    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    # normalize to 0-255
    heatmap = np.uint8(255 * heatmap)
    return heatmap


def overlay_heatmap_on_image(orig_img, heatmap):
    # orig_img: PIL Image RGB
    # heatmap: 2D numpy uint8
    import numpy as _np

    heat = Image.fromarray(heatmap).resize(orig_img.size)
    heat_np = np.array(heat).astype('float32') / 255.0

    # create an RGBA red map
    red = np.zeros((heat_np.shape[0], heat_np.shape[1], 3), dtype=np.float32)
    red[:, :, 0] = 1.0
    # alpha channel from heatmap
    alpha = np.expand_dims(heat_np, axis=2)

    orig = np.array(orig_img).astype('float32') / 255.0
    overlay = (1 - alpha * 0.6) * orig + (alpha * 0.6) * red
    overlay = np.clip(overlay * 255.0, 0, 255).astype('uint8')
    return Image.fromarray(overlay)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image' in request"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    x = prepare_image(img, target_size)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    class_name = CLASSES[idx] if idx < len(CLASSES) else str(idx)
    confidence = float(preds[idx])
    # map to commercial quality
    if idx in (0, 1):
        quality = "Premium"
        if idx == 0:
            quality_reason = "Premium (si no presenta daños)"
        else:
            quality_reason = "Premium"
        development_index = 1
    else:
        quality = "Rechazo"
        if idx == 2:
            quality_reason = "Banano en mal estado o podrido"
        else:
            quality_reason = "Banano inmaduro, aún no apto para exportación"
        development_index = 2

    return jsonify({
        "cultivo": "Banano",
        "clasificacion": class_name,
        "label_id": idx,
        "quality": quality,
        "quality_reason": quality_reason,
        "indice_desarrollo": development_index,
        "confianza": round(confidence * 100, 2),
        "probs": [float(p) for p in preds],
        "labels": CLASSES
    })



@app.route('/explain', methods=['POST'])
def explain():
    if 'image' not in request.files:
        return jsonify({"error": "No file part 'image' in request"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    x = prepare_image(img, target_size)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))

    try:
        heatmap = make_gradcam(x, model, idx)
        overlay = overlay_heatmap_on_image(img, heatmap)
        buf = io.BytesIO()
        overlay.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Grad-CAM failed: {e}"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
