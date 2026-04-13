from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum

class Role(enum.Enum):
    patient = "patient"
    doctor = "doctor"

class Status(enum.Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"
    completed = "completed"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(Enum(Role))
    name = Column(String)
    phone = Column(String, nullable=True)
    bio = Column(Text, nullable=True)

    detections = relationship("Detection", back_populates="user")
    consultations_as_patient = relationship("Consultation", foreign_keys="Consultation.patient_id", back_populates="patient")
    consultations_as_doctor = relationship("Consultation", foreign_keys="Consultation.doctor_id", back_populates="doctor")
    requests = relationship("ConsultationRequest", foreign_keys="ConsultationRequest.patient_id", back_populates="patient")

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String)
    disease = Column(String)
    confidence = Column(Float)
    severity = Column(String)  # e.g., mild, moderate, severe
    recommendations = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="detections")

class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime)
    status = Column(Enum(Status), default=Status.pending)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("User", foreign_keys=[patient_id], back_populates="consultations_as_patient")
    doctor = relationship("User", foreign_keys=[doctor_id], back_populates="consultations_as_doctor")

class ConsultationRequest(Base):
    __tablename__ = "consultation_requests"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime)
    notes = Column(Text, nullable=True)
    scan_id = Column(Integer, ForeignKey("detections.id"), nullable=True)
    status = Column(Enum(Status), default=Status.pending)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("User", foreign_keys=[patient_id], back_populates="requests")
    doctor = relationship("User", foreign_keys=[doctor_id])
    scan = relationship("Detection")