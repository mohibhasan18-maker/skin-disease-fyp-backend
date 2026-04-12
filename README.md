# Backend Project

FastAPI backend for skin disease detection with user authentication, patient and doctor portals.

## Prerequisites
- Python 3.8+
- PostgreSQL
- pip

## Setup
1. Install PostgreSQL and create a database named `skin_detection` with user `user` and password `password`. (Or update DATABASE_URL in database.py)

2. Create a virtual environment (optional):

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run
1. Start the application:

```powershell
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000/api`

2. Seed sample data:

```powershell
python seed.py
```

## API Documentation
- Interactive docs: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Sample Users
- Patient: patient1@example.com / password
- Doctor: doctor1@example.com / password

## Files
- `main.py` — Main application
- `models.py` — Database models
- `auth.py` — Authentication utilities
- `database.py` — Database connection
- `seed.py` — Sample data seeder
- `requirements.txt` — Python dependencies
- `final_model.h5` — ML model file
