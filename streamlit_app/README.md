Minimal Streamlit frontend for the Colorizer project

How to run locally:

1. Create and activate a virtual environment (Windows cmd.exe):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r streamlit_app\requirements.txt
```

2. From the repo root run:

```cmd
streamlit run streamlit_app\app.py
```

The app will open in your browser. Upload a grayscale image and click "Colorize". The app loads the model from `training/runs/best.pt` (ensure the checkpoint is present or adjust the path).