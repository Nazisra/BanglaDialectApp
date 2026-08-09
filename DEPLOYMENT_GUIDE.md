# Render.com Deployment Guide

## 📋 Prerequisites
- GitHub account (with your code pushed)
- Render.com account (free)

## 🚀 Deployment Steps

### Step 1: Push Code to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Bangla Dialect App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/BanglaDialectApp.git
git push -u origin main
```

### Step 2: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Authorize GitHub access

### Step 3: Deploy on Render
1. Go to Dashboard → **New** → **Web Service**
2. Connect your GitHub repository
3. Select: `BanglaDialectApp`
4. Fill in these details:
   - **Name**: `bangla-dialect-app`
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 60`
5. **Plan**: Free (recommended for testing)
6. Click **Deploy Web Service**

### Step 4: Wait for Deployment
- Takes 2-3 minutes
- Check logs for any errors
- Your app URL will be: `https://bangla-dialect-app.onrender.com`

## ⚠️ Important Notes

### Model Files
Your model files (`roberta_bangla_v1/`) are in `.gitignore` and won't be pushed to GitHub.

**Two Options:**

#### Option A: Upload Model to Cloud Storage (Recommended for Free Tier)
1. Upload `roberta_bangla_v1/` folder to:
   - Google Drive / AWS S3 / Hugging Face
2. Set environment variable on Render:
   - Go to Service Settings → Environment
   - Add: `MODEL_SAFETENSORS_URL=https://your-model-url/model.safetensors`
3. App will auto-download on first run

#### Option B: Push Model to GitHub (If < 100MB)
```bash
# Only if your model files are small
git add roberta_bangla_v1/
git push
```

### Render Free Tier Limits
- ⏱️ Auto-spins down after 15 min of inactivity
- 🔧 Takes time to wake up on first request
- 💾 Limited memory (512MB)
- ❌ No persistent storage

### For Production
Upgrade to Paid Plan ($7+/month):
- ✅ Always-on servers
- ✅ More memory/CPU
- ✅ Better performance

## 🔧 Environment Variables
If needed, add in Render Dashboard → Service → Environment:
```
PORT=10000
FLASK_ENV=production
MODEL_SAFETENSORS_URL=https://your-url/model.safetensors
```

## ✅ Test Your App
After deployment:
```bash
curl https://bangla-dialect-app.onrender.com/
# Should return the HTML page
```

## 📱 API Usage
```bash
curl -X POST https://bangla-dialect-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "আপনে কইতাছেন"}'
```

## ❓ Troubleshooting

### Model files not found
- Check environment variable is set correctly
- Verify download URL is public

### Service keeps spinning down
- Upgrade to paid plan
- Or keep it warm with periodic requests

### Build fails
- Check `requirements.txt` has correct versions
- Check Python version is 3.11

## 📞 Support
- Render Docs: https://docs.render.com
- Python Deployment: https://docs.render.com/deploy-python

---
**Happy Deploying!** 🎉
