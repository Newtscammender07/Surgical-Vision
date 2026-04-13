# Surgical Instrument AR Monitor

An AI-powered augmented reality dashboard for real-time surgical instrument detection and monitoring. This system uses YOLOv8 to identify surgical tools (clamps, forceps, parker) and provides a professional Streamlit interface for visualization and usage tracking.

## 🚀 Features
- **Real-time Detection**: Multi-tool detection with high precision using YOLOv8.
- **AR Dashboard**: Interactive Streamlit UI for monitoring live surgical feeds.
- **Instrument Tracking**: Automated counting and usage logging.
- **Training Pipeline**: Custom training scripts to fine-tune models on surgical datasets.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd surgical_ar_github
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Setup

This project uses a surgical instrument dataset. You can download it directly using the provided script:

```bash
python download_dataset.py
```

After downloading, ensure the data is formatted correctly:
```bash
python format_dataset.py
```

## 🖥️ Usage

To start the AR monitoring dashboard:
```bash
python app.py
```
Or run the provided batch script on Windows:
```bash
start_app.bat
```

## 🏋️ Training
To train the model on your own data:
```bash
python train_yolov8.py
```

## 📜 License
This project is for academic/research purposes.
