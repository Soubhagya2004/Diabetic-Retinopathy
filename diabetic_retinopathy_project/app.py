from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from train_model import focal_loss, build_model

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
MODEL_PATH = ROOT / "model" / "dr_model.h5"
IMG_SIZE = (224, 224)

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

app = Flask(__name__)
UPLOADS.mkdir(parents=True, exist_ok=True)

# Compatibility shim for models that reference an operation-named layer 'TrueDivide'.
# When models are serialized, some TensorFlow ops may be represented as layer-like
# objects; providing a small passthrough layer here lets `load_model` resolve that
# name via `custom_objects` instead of failing with "Unknown layer: 'TrueDivide'".
class TrueDivide(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        # Accept positional args that may represent a divisor (e.g. 127.5)
        super().__init__(**kwargs)
        self.divisor = None
        if args:
            # if first positional arg is a number, treat as divisor
            first = args[0]
            if isinstance(first, (int, float)):
                self.divisor = float(first)
        # also support keyword form if present
        if "value" in kwargs and isinstance(kwargs.get("value"), (int, float)):
            self.divisor = float(kwargs.get("value"))

    def call(self, inputs):
        if self.divisor is None:
            return inputs
        return inputs / self.divisor


def load_model_if_available():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100:
        try:
                # Try loading with the custom loss used during training and small
                # compatibility shims for any operation-named layers.
                try:
                    custom_objs = {"focal_loss": focal_loss, "TrueDivide": TrueDivide}
                    model = tf.keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objs)
                    print(f"Loaded model (with custom_objects) from {MODEL_PATH}")
                    return model
                except Exception as e_custom:
                    print("Loading with custom_objects failed, trying compile=False fallback:", e_custom)
                try:
                    # include the same custom objects when using compile=False so
                    # unknown layers are still resolvable during deserialization
                    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False, custom_objects=custom_objs)
                    print(f"Loaded model (compile=False) from {MODEL_PATH}")
                    return model
                except Exception as e2:
                    print("Failed to load model with compile=False:", e2)
                    # Final fallback: try to reconstruct the architecture and load weights
                    try:
                        print("Attempting to build model architecture and load weights...")
                        model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                        model.load_weights(str(MODEL_PATH))
                        print(f"Loaded weights into rebuilt model from {MODEL_PATH}")
                        return model
                    except Exception as e3:
                        print("Failed to load weights into rebuilt model:", e3)
                        return None
        except Exception as e:
            print("Failed to load model:", e)
            return None
    else:
        print("Model file missing or invalid. Run `python train_model.py` to create it.")
        return None


MODEL = load_model_if_available()


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    # Normalize to [0,1] as training/diagnose code did; the model
    # includes a MobileNetV2 preprocessing layer, so training used
    # inputs in 0-1 before that layer.
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOADS), filename)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        filename = secure_filename(file.filename)
        save_path = UPLOADS / filename
        file.save(save_path)

        if MODEL is None:
            prediction = "Model not available. Run `python train_model.py` and restart app.`"
        else:
            x = preprocess_image(save_path)
            probs = MODEL.predict(x)[0]
            idx = int(np.argmax(probs))
            prediction = CLASSES[idx]
            confidence = float(probs[idx])

    return render_template("index.html", prediction=prediction, confidence=confidence, filename=filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
