# Backend Project

FastAPI backend for skin disease detection with user authentication, patient and doctor portals.

## Prerequisites
- Python 3.8+
- PostgreSQL
- pip

## Setup
1. Install PostgreSQL.

2. Create a PostgreSQL role and database for the app.

   Open a terminal and run `psql` as the PostgreSQL superuser (usually `postgres`):

   ```sql
   CREATE USER user WITH PASSWORD 'password';
   CREATE DATABASE skin_detection OWNER user;
   GRANT ALL PRIVILEGES ON DATABASE skin_detection TO user;
   \q
   ```

   If you want a different user, database name, or password, update the `DATABASE_URL` environment variable instead of using these defaults.

3. Configure the database connection.

   The app uses the `DATABASE_URL` environment variable when present. Example values:

   - Windows PowerShell:
     ```powershell
     $env:DATABASE_URL = "postgresql://user:password@localhost:5432/skin_detection"
     ```

   - Linux/macOS Bash:
     ```bash
     export DATABASE_URL="postgresql://user:password@localhost:5432/skin_detection"
     ```

   If `DATABASE_URL` is not set, the app falls back to a local SQLite database file at `./skin_detection.db`.

4. Create a virtual environment (optional):

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

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

## Render Deployment
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Set `SECRET_KEY` in Render environment variables.
- If `final_modelv2.h5` is not committed, set `MODEL_URL` to a direct download URL for the model file.
- Optional: set `FRONTEND_URLS` if you need to allow more frontend domains. The default allows `http://localhost:3000` and `https://skin-disease-rho.vercel.app`.
- The deployed model file is `final_modelv2.h5`.

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
- `final_modelv2.h5` — ML model file
