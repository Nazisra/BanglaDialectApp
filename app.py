import os
import urllib.request
from flask import Flask, jsonify, render_template, request

# Optional ONNX Runtime path (for low-memory inference without torch)
np = None
ort = None

# Lazy imports - import only when needed to avoid startup hangs
torch = None
AutoTokenizer = None
AutoModelForSequenceClassification = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "roberta_bangla_v1")
MODEL_WEIGHTS_PATH = os.path.join(MODEL_PATH, "model.safetensors")
ONNX_MODEL_INT8_PATH = os.path.join(MODEL_PATH, "model-int8.onnx")
ONNX_MODEL_FP32_PATH = os.path.join(MODEL_PATH, "model.onnx")

# আপনার স্ক্রিনশট (le.classes_) অনুযায়ী একদম সঠিক সিরিয়াল
classes = [
    'Barisal',          # 0
    'Chittagong',       # 1
    'Comilla',          # 2
    'Dhaka',            # 3
    'Khulna',           # 4
    'Mymensingh',       # 5
    'Noakhali',         # 6
    'Rajshahi',         # 7
    'Rangpur',          # 8
    'Standard_Bangla',  # 9
    'Sylhet'            # 10
]

tokenizer = None
model = None
model_load_error = None

onnx_tokenizer = None
onnx_session = None
onnx_load_error = None


def _softmax_np(logits):
    # logits: (batch, num_labels)
    exp_shifted = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


def load_onnx_model():
    global onnx_tokenizer, onnx_session, onnx_load_error, np, ort

    if onnx_tokenizer is not None and onnx_session is not None:
        return True

    onnx_path = None
    if os.path.exists(ONNX_MODEL_INT8_PATH):
        onnx_path = ONNX_MODEL_INT8_PATH
    elif os.path.exists(ONNX_MODEL_FP32_PATH):
        onnx_path = ONNX_MODEL_FP32_PATH
    else:
        onnx_load_error = "ONNX model file not found"
        return False

    try:
        if np is None:
            import numpy as numpy_module
            np = numpy_module

        if ort is None:
            import onnxruntime as ort_module
            ort = ort_module

        # Tokenizer can be loaded without torch
        from transformers import AutoTokenizer as AT

        print(f"Loading ONNX model from: {onnx_path}...")
        onnx_tokenizer = AT.from_pretrained(MODEL_PATH, local_files_only=True)

        sess_options = ort.SessionOptions()
        # Keep memory/threads low for small instances
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        onnx_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        onnx_load_error = None
        print("ONNX model loaded successfully")
        return True
    except Exception as exc:
        onnx_tokenizer = None
        onnx_session = None
        onnx_load_error = str(exc)
        print(f"❌ Error loading ONNX model: {exc}")
        return False


def load_model():
    global tokenizer, model, model_load_error, torch, AutoTokenizer, AutoModelForSequenceClassification

    if tokenizer is not None and model is not None:
        return True

    try:
        # Optional: download weights at runtime (useful for cloud deploys)
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            weights_url = os.environ.get("MODEL_SAFETENSORS_URL", "").strip()
            if weights_url:
                os.makedirs(MODEL_PATH, exist_ok=True)
                print("Model weights missing; downloading model.safetensors...")
                urllib.request.urlretrieve(weights_url, MODEL_WEIGHTS_PATH)
                print("Downloaded model.safetensors")

        # Import on demand
        if torch is None:
            import torch as torch_module
            torch = torch_module
        
        if AutoTokenizer is None:
            from transformers import AutoTokenizer as AT, AutoModelForSequenceClassification as AMFC
            AutoTokenizer = AT
            AutoModelForSequenceClassification = AMFC
        
        print(f"Loading model from: {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        model.eval()
        model_load_error = None
        print("Model and tokenizer loaded successfully")
        return True
    except Exception as exc:
        tokenizer = None
        model = None
        model_load_error = str(exc)
        print(f"❌ Error loading model: {exc}")
        return False


def normalize_text(text):
    return " ".join(text.lower().split())


def heuristic_predict(text):
    normalized = normalize_text(text)

    keyword_map = {
        "Barisal": ["আপনে", "কইতাছেন", "হইছে", "কইরা", "যাইতেছেন", "করতাছেন"],
        "Chittagong": ["তুই", "গইলাম", "যাইতেছি", "কইলাম", "নাইকা", "হইতেছে"],
        "Comilla": ["এইডা", "কিতা", "কইতাছ", "হইলো", "তর", "কয়"],
        "Dhaka": ["তুমি", "করছ", "কেমন", "আছি", "আছো", "আসো"],
        "Khulna": ["তোরা", "কইতাছ", "হইতেসে", "করতেছি", "নাইরে"],
        "Mymensingh": ["আমাগো", "তাগো", "কইরা", "যাইতাছি", "হইল", "আসতেছে"],
        "Noakhali": ["আপনের", "কইতাছেন", "হইছে", "নাই", "করতাছেন", "যাইতাছেন"],
        "Rajshahi": ["কইরেন", "করতাছেন", "হইবো", "যাইতেছে", "নাইতো", "খাইছেন"],
        "Rangpur": ["কিতা", "কইতাছ", "হইল", "যাইতেছে", "আছই", "করতাছি"],
        "Standard_Bangla": ["তুমি", "আপনি", "কেমন", "আছেন", "করছেন", "আজকে"],
        "Sylhet": ["কই", "কিতা", "যাইতেছি", "গইলাম", "আঁ", "হইলেন"],
    }

    scores = {label: 0 for label in classes}
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in normalized:
                scores[label] += 1

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score == 0:
        return "Standard_Bangla", 0.52

    confidence = min(0.95, 0.45 + (best_score * 0.12))
    return best_label, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global torch
    payload = request.get_json(silent=True) or {}

    try:
        text = payload.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Prefer ONNX Runtime path (much lower RAM than torch)
        if load_onnx_model() and onnx_session is not None and onnx_tokenizer is not None and np is not None:
            enc = onnx_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
            )

            feed = {}
            input_names = {i.name for i in onnx_session.get_inputs()}
            if "input_ids" in input_names:
                feed["input_ids"] = np.array([enc["input_ids"]], dtype=np.int64)
            if "attention_mask" in input_names and "attention_mask" in enc:
                feed["attention_mask"] = np.array([enc["attention_mask"]], dtype=np.int64)
            if "token_type_ids" in input_names and "token_type_ids" in enc:
                feed["token_type_ids"] = np.array([enc["token_type_ids"]], dtype=np.int64)

            outputs = onnx_session.run(None, feed)
            logits = outputs[0]
            probs = _softmax_np(logits)

            predicted_index = int(np.argmax(probs, axis=-1)[0])
            predicted_confidence = float(probs[0, predicted_index])

            top3 = []
            try:
                top_indices = np.argsort(-probs[0])[: min(3, probs.shape[-1])]
                for idx in top_indices.tolist():
                    top3.append({
                        "index": int(idx),
                        "raw_label": f"LABEL_{int(idx)}",
                        "dialect": classes[int(idx)] if 0 <= int(idx) < len(classes) else None,
                        "prob": float(probs[0, int(idx)]),
                    })
            except Exception:
                top3 = []

            return jsonify({
                "dialect": classes[predicted_index] if 0 <= predicted_index < len(classes) else None,
                "confidence": f"{predicted_confidence * 100:.2f}%",
                "source": "onnx",
                "label_index": predicted_index,
                "raw_label": f"LABEL_{predicted_index}",
                "top3": top3,
            })

        # Fallback to torch model path if available
        if load_model() and torch is not None:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_idx = torch.max(probs, dim=1)

            predicted_index = int(predicted_idx.item())
            predicted_confidence = float(confidence.item())

            raw_label = None
            try:
                id2label = getattr(getattr(model, "config", None), "id2label", None)
                if isinstance(id2label, dict):
                    raw_label = id2label.get(predicted_index)
                    if raw_label is None:
                        raw_label = id2label.get(str(predicted_index))
            except Exception:
                raw_label = None

            topk = []
            try:
                top_values, top_indices = torch.topk(probs[0], k=min(3, probs.shape[-1]))
                for value, index_tensor in zip(top_values.tolist(), top_indices.tolist()):
                    index_int = int(index_tensor)
                    item_raw = None
                    try:
                        id2label = getattr(getattr(model, "config", None), "id2label", None)
                        if isinstance(id2label, dict):
                            item_raw = id2label.get(index_int)
                            if item_raw is None:
                                item_raw = id2label.get(str(index_int))
                    except Exception:
                        item_raw = None

                    item_dialect = classes[index_int] if 0 <= index_int < len(classes) else None
                    topk.append({
                        "index": index_int,
                        "raw_label": item_raw,
                        "dialect": item_dialect,
                        "prob": float(value),
                    })
            except Exception:
                topk = []

            return jsonify({
                "dialect": classes[predicted_index] if 0 <= predicted_index < len(classes) else None,
                "confidence": f"{predicted_confidence * 100:.2f}%",
                "source": "model",
                "label_index": predicted_index,
                "raw_label": raw_label,
                "top3": topk,
            })

        fallback_dialect, fallback_confidence = heuristic_predict(text)
        return jsonify({
            "dialect": fallback_dialect,
            "confidence": f"{fallback_confidence * 100:.2f}%",
            "source": "fallback",
            "warning": "Model unavailable; using heuristic prediction.",
            "onnx_error": onnx_load_error,
            "torch_error": model_load_error,
        })
    except Exception as exc:
        fallback_dialect, fallback_confidence = heuristic_predict(payload.get("text", ""))
        return jsonify({
            "dialect": fallback_dialect,
            "confidence": f"{fallback_confidence * 100:.2f}%",
            "source": "fallback",
            "warning": f"Prediction fallback used after error: {exc}"
        })

if __name__ == "__main__":
    print("🚀 Starting Flask app at http://127.0.0.1:5000")
    print("📝 Enter text to classify dialect. Press Ctrl+C to stop.")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)