from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from database import engine, get_db, Base
from models import User, Detection, Consultation, ConsultationRequest, Role, Status
from auth import get_current_user, create_access_token, verify_password, get_password_hash

# Create tables
Base.metadata.create_all(bind=engine)

# -----------------------------
# App Setup
# -----------------------------
app = FastAPI(
    title="Skin Disease Detection API",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url="/api/redoc"
)
router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Constants
# -----------------------------
IMAGE_SIZE = 224
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASS_NAMES = [
    "Acne",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Eczema",
    "Non Skin",
    "Normal Skin",
    "Psoriasis",
]

CONFIDENCE_THRESHOLD = 0.75
GAP_THRESHOLD = 0.2

# -----------------------------
# Model Path
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_modelv2.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("final_model.h5 not found!")

# -----------------------------
# Load Model (ONCE)
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# -----------------------------
# Pydantic Models
# -----------------------------
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str
    role: Optional[Role] = Role.patient
    phone: Optional[str] = None
    bio: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    name: str
    phone: Optional[str]
    bio: Optional[str]

    class Config:
        orm_mode = True

class ProfileUpdate(BaseModel):
    name: Optional[str]
    phone: Optional[str]
    bio: Optional[str]

class DetectionResponse(BaseModel):
    id: int
    disease: str
    confidence: float
    severity: str
    recommendations: str
    created_at: datetime

    class Config:
        orm_mode = True

class ConsultationResponse(BaseModel):
    id: int
    doctor_id: int
    doctor_name: str
    date: datetime
    status: str
    notes: Optional[str]

class RequestCreate(BaseModel):
    doctor_id: int
    date: datetime
    notes: Optional[str]
    scan_id: Optional[int]

class DoctorResponse(BaseModel):
    id: int
    name: str
    bio: Optional[str]

class DashboardStats(BaseModel):
    total_scans: int
    upcoming_consultations: int
    recent_activity: List[dict]

class RequestResponse(BaseModel):
    id: int
    patient_id: int
    patient_name: str
    date: datetime
    notes: Optional[str]
    scan_id: Optional[int]
    status: str

class NotesUpdate(BaseModel):
    notes: str

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(image)
    image_array = tf.keras.applications.efficientnet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def get_severity_and_recommendations(disease: str, confidence: float):
    if disease == "Normal Skin":
        return "None", "No action needed. Maintain healthy skin care routine."
    elif disease in ["Acne", "Eczema"]:
        return "Mild", "Consult a dermatologist for topical treatments. Avoid irritants."
    elif disease == "Psoriasis":
        return "Moderate", "Seek medical advice for appropriate therapy. Moisturize regularly."
    elif disease in ["Atopic Dermatitis", "Basal Cell Carcinoma"]:
        return "Severe", "Immediate medical consultation required. Do not delay."
    else:
        return "Unknown", "Consult a healthcare professional for proper diagnosis."

# -----------------------------
# Routes
# -----------------------------
@router.get("/")
def health_check():
    return {"status": "API is running"}

# Authentication
@router.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role.value,
            "name": user.name,
            "phone": user.phone,
            "bio": user.bio
        }
    }

@router.post("/auth/signup", response_model=TokenResponse)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=request.email,
        password_hash=get_password_hash(request.password),
        role=request.role,
        name=request.name,
        phone=request.phone,
        bio=request.bio
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role.value,
            "name": user.name,
            "phone": user.phone,
            "bio": user.bio
        }
    }

@router.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value,
        "name": current_user.name,
        "phone": current_user.phone,
        "bio": current_user.bio,
    }

@router.put("/users/profile", response_model=UserResponse)
def update_profile(request: ProfileUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if request.name:
        current_user.name = request.name
    if request.phone:
        current_user.phone = request.phone
    if request.bio:
        current_user.bio = request.bio
    db.commit()
    db.refresh(current_user)
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value,
        "name": current_user.name,
        "phone": current_user.phone,
        "bio": current_user.bio,
    }

# Patient Portal
@router.get("/patient/dashboard")
def patient_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.patient:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    total_scans = db.query(Detection).filter(Detection.user_id == current_user.id).count()
    upcoming_consultations = db.query(Consultation).filter(
        Consultation.patient_id == current_user.id,
        Consultation.status == Status.accepted,
        Consultation.date > datetime.utcnow()
    ).count()
    
    recent_scans = db.query(Detection).filter(Detection.user_id == current_user.id).order_by(Detection.created_at.desc()).limit(5).all()
    recent_activity = [{"type": "scan", "disease": s.disease, "date": s.created_at} for s in recent_scans]
    
    return {
        "total_scans": total_scans,
        "upcoming_consultations": upcoming_consultations,
        "recent_activity": recent_activity
    }

@router.post("/patient/detection/analyze")
async def analyze_detection(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.patient:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        predictions = model.predict(input_tensor)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        sorted_preds = np.sort(predictions)
        gap = float(sorted_preds[-1] - sorted_preds[-2])

        if confidence < CONFIDENCE_THRESHOLD or gap < GAP_THRESHOLD:
            disease = "Uncertain"
        else:
            disease = CLASS_NAMES[predicted_index]

        severity, recommendations = get_severity_and_recommendations(disease, confidence)

        # Save image
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # Save to db
        detection = Detection(
            user_id=current_user.id,
            image_path=filepath,
            disease=disease,
            confidence=confidence,
            severity=severity,
            recommendations=recommendations
        )
        db.add(detection)
        db.commit()
        db.refresh(detection)

        return {
            "id": detection.id,
            "disease": disease,
            "confidence": round(confidence, 4),
            "confidences": {class_name: round(float(pred), 4) for class_name, pred in zip(CLASS_NAMES, predictions)},
            "severity": severity,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patient/detection/history", response_model=List[DetectionResponse])
def detection_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.patient:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    detections = db.query(Detection).filter(Detection.user_id == current_user.id).order_by(Detection.created_at.desc()).all()
    return detections

@router.get("/patient/consultations", response_model=List[ConsultationResponse])
def patient_consultations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.patient:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    consultations = db.query(Consultation).filter(Consultation.patient_id == current_user.id).order_by(Consultation.date.desc()).all()
    result = []
    for c in consultations:
        result.append({
            "id": c.id,
            "doctor_id": c.doctor_id,
            "doctor_name": c.doctor.name,
            "date": c.date,
            "status": c.status.value,
            "notes": c.notes
        })
    return result

@router.post("/patient/consultations/request")
def create_request(request: RequestCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.patient:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    consultation_request = ConsultationRequest(
        patient_id=current_user.id,
        doctor_id=request.doctor_id,
        date=request.date,
        notes=request.notes,
        scan_id=request.scan_id
    )
    db.add(consultation_request)
    db.commit()
    db.refresh(consultation_request)
    return {"id": consultation_request.id, "message": "Request sent"}

# Doctor Portal
@router.get("/doctor/dashboard")
def doctor_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.doctor:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    pending_requests = db.query(ConsultationRequest).filter(
        ConsultationRequest.doctor_id == current_user.id,
        ConsultationRequest.status == Status.pending
    ).count()
    
    today = datetime.utcnow().date()
    start_of_today = datetime(today.year, today.month, today.day)
    end_of_today = start_of_today + timedelta(days=1)
    today_appointments = db.query(Consultation).filter(
        Consultation.doctor_id == current_user.id,
        Consultation.date >= start_of_today,
        Consultation.date < end_of_today,
        Consultation.status == Status.accepted
    ).count()
    
    total_patients = db.query(Consultation).filter(Consultation.doctor_id == current_user.id).distinct(Consultation.patient_id).count()
    
    return {
        "pending_requests": pending_requests,
        "today_appointments": today_appointments,
        "total_patients": total_patients
    }

@router.get("/doctor/requests", response_model=List[RequestResponse])
def doctor_requests(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.doctor:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    requests = db.query(ConsultationRequest).filter(ConsultationRequest.doctor_id == current_user.id).order_by(ConsultationRequest.created_at.desc()).all()
    result = []
    for r in requests:
        result.append({
            "id": r.id,
            "patient_id": r.patient_id,
            "patient_name": r.patient.name,
            "date": r.date,
            "notes": r.notes,
            "scan_id": r.scan_id,
            "status": r.status.value
        })
    return result

@router.put("/doctor/requests/{request_id}/status")
def update_request_status(request_id: int, status: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.doctor:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    request_obj = db.query(ConsultationRequest).filter(
        ConsultationRequest.id == request_id,
        ConsultationRequest.doctor_id == current_user.id
    ).first()
    if not request_obj:
        raise HTTPException(status_code=404, detail="Request not found")
    
    if status not in ["accepted", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    request_obj.status = Status(status)
    if status == "accepted":
        consultation = Consultation(
            patient_id=request_obj.patient_id,
            doctor_id=current_user.id,
            date=request_obj.date,
            status=Status.accepted
        )
        db.add(consultation)
    
    db.commit()
    return {"message": f"Request {status}"}

@router.get("/doctor/consultations", response_model=List[ConsultationResponse])
def doctor_consultations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.doctor:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    consultations = db.query(Consultation).filter(Consultation.doctor_id == current_user.id).order_by(Consultation.date.desc()).all()
    result = []
    for c in consultations:
        result.append({
            "id": c.id,
            "doctor_id": c.doctor_id,
            "doctor_name": current_user.name,
            "date": c.date,
            "status": c.status.value,
            "notes": c.notes
        })
    return result

@router.post("/doctor/consultations/{consultation_id}/notes")
def add_consultation_notes(consultation_id: int, request: NotesUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != Role.doctor:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    consultation = db.query(Consultation).filter(
        Consultation.id == consultation_id,
        Consultation.doctor_id == current_user.id
    ).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    
    consultation.notes = request.notes
    consultation.status = Status.completed
    db.commit()
    return {"message": "Notes added"}

# General
@router.get("/doctors", response_model=List[DoctorResponse])
def list_doctors(db: Session = Depends(get_db)):
    doctors = db.query(User).filter(User.role == Role.doctor).all()
    return [{"id": d.id, "name": d.name, "bio": d.bio} for d in doctors]

app.include_router(router)

