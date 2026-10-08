# TriVIA — Traffic Vision & Intelligent Analytics

## Run in VS Code (without a virtual environment)

Open this folder in VS Code and use the integrated terminal.

### 1. Check Python
```powershell
python --version
```

Python 3.11 or 3.12 is recommended for the ML packages.

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

If Windows uses `py` instead of `python`:
```powershell
py -m pip install -r requirements.txt
```

### 3. Start the application
```powershell
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Project structure
```text
TriVIA/
├── app.py
├── requirements.txt
├── yolo11n.pt
├── run_trivia.bat
├── README.md
├── data/
│   └── input/
├── outputs/
└── src/
    ├── rcnn_detector.py
    ├── result_fusion.py
    ├── traffic_analyzer.py
    ├── interpretation_engine.py
    ├── vehicle_counter.py
    ├── vehicle_tracker.py
    ├── video_processor.py
    ├── visualization.py
    └── yolo_detector.py
```

## Input
Upload a traffic video from the Streamlit interface. Short 5–15 second videos are recommended for a quick demo.

The system combines YOLO and Faster R-CNN detections, tracks vehicles, counts traffic, compares model detections, and generates traffic analysis.

## Run from the VS Code terminal
```powershell
streamlit run app.py
```

No virtual environment is required.
