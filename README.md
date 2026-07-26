# 🔎 AI Vision Scanner

An open-source web app that detects **text (OCR)** and **objects** in any uploaded photo — built with [Streamlit](https://streamlit.io), [EasyOCR](https://github.com/JaidedAI/EasyOCR), and [YOLOv8](https://github.com/ultralytics/ultralytics).

No API keys, no cloud services, no paid tools — everything runs locally or on Streamlit Community Cloud for free.

## ✨ Features

- 📝 **Text detection** — extract readable text from photos, receipts, signs, documents
- 📦 **Object detection** — identify and locate 80+ common object classes (YOLOv8)
- 🎚️ Adjustable confidence threshold
- 🎨 Clean, dark-themed, mobile-friendly UI
- 📥 Download results as `.txt`
- 🔧 Debug view showing the OCR pre-processing pipeline (grayscale, blur, adaptive threshold, deskew)

## 🚀 Quick Start (local)

```bash
git clone https://github.com/<your-username>/ai-vision-scanner.git
cd ai-vision-scanner
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

> **Note:** the first run downloads the EasyOCR and YOLOv8 model weights (~100MB total). This only happens once — they're cached afterward.

> **Python version:** use Python 3.11 or 3.12. Some dependencies (numpy, opencv, torch) may not yet have prebuilt wheels for the newest Python releases.

## ☁️ Deploy for free on Streamlit Community Cloud

1. Push this project to your own public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo → set the main file to `app.py`
4. Click **Deploy**

That's it — you'll get a public URL like `https://your-app-name.streamlit.app`.

## 📁 Project Structure

```
ai-vision-scanner/
├── app.py                   # Streamlit UI + orchestration
├── requirements.txt
├── .streamlit/
│   └── config.toml          # custom dark theme
└── utils/
    ├── preprocess.py        # grayscale, blur, threshold, deskew
    ├── ocr_detect.py        # EasyOCR text detection
    └── object_detect.py     # YOLOv8 object detection
```

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI / Web framework | Streamlit |
| Text recognition | EasyOCR |
| Object detection | YOLOv8-nano (Ultralytics) |
| Image processing | OpenCV |

## 🤝 Contributing

Pull requests are welcome! Ideas for contributions:
- Add more languages to EasyOCR (`utils/ocr_detect.py` → `easyocr.Reader([...])`)
- Swap in a larger YOLO model for higher accuracy (`yolov8s.pt`, `yolov8m.pt`)
- Add batch upload support
- Add PDF input support

👨‍💻 Author
ABHISHEK THUVVARA

B.Tech Artificial Intelligence Student

LinkedIn:https://www.linkedin.com/in/thuvvara-abhishek-01768b362/

github:-https://github.com/abhishekthuvvara

📜 License This project is created for educational and internship purposes under the DecodeLabs AI Internship Program.
