# Bangla Dialect Classification App

একটি RoBERTa-based বাংলা ডায়ালেক্ট ক্লাসিফাইজেশন অ্যাপ্লিকেশন।

## 🎯 Features

- ✅ RoBERTa মডেল দিয়ে বাংলা ডায়ালেক্ট শনাক্তকরণ
- ✅ 11টি ডায়ালেক্ট সাপোর্ট:
  - Barisal, Chittagong, Comilla, Dhaka, Khulna
  - Mymensingh, Noakhali, Rajshahi, Rangpur
  - Standard Bangla, Sylhet

- ✅ ONNX Runtime সাপোর্ট (কম মেমোরি ব্যবহার)
- ✅ Fallback হিউরিস্টিক প্রেডিকশন
- ✅ Web UI + REST API

## 📦 Installation

### Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/BanglaDialectApp.git
cd BanglaDialectApp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model
Model ফাইল প্রয়োজন: `roberta_bangla_v1/`
- এটি রেপো তে অন্তর্ভুক্ত নয় (`.gitignore` এ আছে)
- [আপনার মডেল সোর্স থেকে ডাউনলোড করুন]

## 🚀 Run Locally

```bash
python app.py
# Opens at http://localhost:5000
```

## 📡 API Endpoints

### Home Page
```
GET /
```

### Predict Dialect
```
POST /predict
Content-Type: application/json

{
  "text": "আপনে কইতাছেন"
}

Response:
{
  "dialect": "Barisal",
  "confidence": "87.45%",
  "source": "onnx",
  "top3": [...]
}
```

## 🌐 Deploy to Render.com

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

Quick start:
1. Push to GitHub
2. Connect to Render.com
3. Deploy!

## 📁 Project Structure

```
BanglaDialectApp/
├── app.py                  # Flask main app
├── wsgi.py                 # WSGI entry point
├── requirements.txt        # Dependencies
├── render.yaml            # Render config
├── Procfile               # Gunicorn config
├── DEPLOYMENT_GUIDE.md    # Deployment steps
├── roberta_bangla_v1/     # Model files
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **ML Model**: Transformers (RoBERTa)
- **Inference**: ONNX Runtime
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Render.com

## 📊 Supported Dialects

| Index | Dialect | Region |
|-------|---------|--------|
| 0 | Barisal | বরিশাল |
| 1 | Chittagong | চট্টগ্রাম |
| 2 | Comilla | কুমিল্লা |
| 3 | Dhaka | ঢাকা |
| 4 | Khulna | খুলনা |
| 5 | Mymensingh | ময়মনসিংহ |
| 6 | Noakhali | নোয়াখালী |
| 7 | Rajshahi | রাজশাহী |
| 8 | Rangpur | রংপুর |
| 9 | Standard_Bangla | মান বাংলা |
| 10 | Sylhet | সিলেট |

## 🔧 Environment Variables

```
PORT=5000                  # Default: 5000
MODEL_SAFETENSORS_URL=...  # For cloud deployment
```

## 📝 Notes

- Model files are large (~500MB+), stored locally
- For cloud deployment, use cloud storage for models
- Free Render tier has 15-min auto-shutdown
- ONNX Runtime is recommended for memory efficiency

## 👨‍💻 Development

```bash
# Run with debug mode (local only)
FLASK_ENV=development python app.py

# Run with Gunicorn
gunicorn app:app
```

## 📄 License

[Your License Here]

## 📧 Contact

[Your Contact Info]

---

**Made with ❤️ for Bangla NLP** 🇧🇩
