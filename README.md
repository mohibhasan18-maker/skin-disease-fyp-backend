# Backend Project

Simple Python backend that uses a Keras model. This repository contains `main.py` which runs the project and the required model files.

## Prerequisites
- Python 3.8+
- pip

## Setup
1. (Optional) Create a virtual environment:

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run
Start the application with:

```powershell
python main.py
```

The script will load model files (if present) such as `model.h5` or `weights.h5` from the project root.

## Files
- `main.py` — application entry point
- `requirements.txt` — Python dependencies
- `model.h5`, `weights.h5` — ML model files (ignored by .gitignore by default)

## Notes
- Large model files are ignored by the provided `.gitignore`. If you want to include a model in the repo, remove the corresponding entry from `.gitignore` before committing.

## License
Add a license if you plan to make this public.
