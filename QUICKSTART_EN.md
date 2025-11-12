# Quick Start Guide

[日本語版](QUICKSTART.md)

## 🚀 How to Run

### GUI Launch (Recommended)

#### Windows (Batch file)
```bash
launch.bat
```
or
```bash
launch.ps1
```

#### Windows (Direct execution)
```bash
venv\Scripts\python.exe modern_gui.py
```

#### Linux/Mac
```bash
./venv/bin/python modern_gui.py
```

#### If virtual environment is activated
```bash
python modern_gui.py
```

---

## 📝 Basic Usage

### Single Image Comparison

1. Click "📁 Select Image 1" to choose the first image
2. Click "📁 Select Image 2" to choose the second image
3. (Optional) Click "🎯 Original Image" to select the low-resolution original
4. Click "🚀 Start Analysis"
5. Click "📂 Open Results Folder" to view results

### Batch Processing

1. Open the "Batch Processing" tab in the GUI
2. Select input and output folders
3. Choose processing count (10/50/100/All)
4. Click "Start Batch Processing"

---

## ⚙️ Command Line Execution

### Single Image Analysis
```bash
venv\Scripts\python.exe advanced_image_analyzer.py image1.png image2.png
```

### With Original Image
```bash
venv\Scripts\python.exe advanced_image_analyzer.py image1.png image2.png --original original.png
```

### Batch Processing
```bash
venv\Scripts\python.exe batch_analyzer.py batch_config.json
```

---

## 🔧 Troubleshooting

### Error: `ModuleNotFoundError`
Virtual environment is not activated. Use the launch methods above.

### Error: `python: command not found`
Specify the Python path directly:
- Windows: `venv\Scripts\python.exe`
- Linux/Mac: `./venv/bin/python`

---

For detailed documentation, see [README.md](README.md).
