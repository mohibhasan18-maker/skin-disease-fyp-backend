from database import SessionLocal, Base, engine
from models import User, Role
from auth import get_password_hash

Base.metadata.create_all(bind=engine)

db = SessionLocal()

patients = [
    {
        "email": "ali.shah@example.com",
        "name": "Ali Shah",
        "role": Role.patient,
        "phone": "+92-300-1234567"
    },
    {
        "email": "ayesha.khan@example.com",
        "name": "Ayesha Khan",
        "role": Role.patient,
        "phone": "+92-301-7654321"
    },
    {
        "email": "sara.ahmed@example.com",
        "name": "Sara Ahmed",
        "role": Role.patient,
        "phone": "+92-302-1122334"
    },
]

doctors = [
    {
        "email": "aamir.khan@example.com",
        "name": "Dr. Aamir Khan",
        "role": Role.doctor,
        "phone": "+92-300-9988776",
        "bio": "Dermatologist with 12 years of experience"
    },
    {
        "email": "fatima.malik@example.com",
        "name": "Dr. Fatima Malik",
        "role": Role.doctor,
        "phone": "+92-301-8877665",
        "bio": "Skin specialist focused on acne and eczema"
    },
    {
        "email": "hassan.raza@example.com",
        "name": "Dr. Hassan Raza",
        "role": Role.doctor,
        "phone": "+92-302-7766554",
        "bio": "Experienced dermatologist for skin cancer and rashes"
    },
    {
        "email": "sana.iqbal@example.com",
        "name": "Dr. Sana Iqbal",
        "role": Role.doctor,
        "phone": "+92-303-6655443",
        "bio": "Dermatologist specializing in psoriasis and dermatitis"
    },
    {
        "email": "naveed.ali@example.com",
        "name": "Dr. Naveed Ali",
        "role": Role.doctor,
        "phone": "+92-304-5544332",
        "bio": "Clinical skin specialist with 8 years practice"
    },
]

all_users = patients + doctors

for user_data in all_users:
    if db.query(User).filter(User.email == user_data["email"]).first():
        continue

    user = User(
        email=user_data["email"],
        password_hash=get_password_hash("123456"),
        role=user_data["role"],
        name=user_data["name"],
        phone=user_data.get("phone"),
        bio=user_data.get("bio")
    )
    db.add(user)

if all_users:
    db.commit()

db.close()

print("Sample users created or already exist.")