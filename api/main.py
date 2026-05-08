# api/main.py
# SenSante API - Assistant pré-diagnostic médical
# Lab 3 - ESP/UCAD - Mohamed Mbaye Ndiaye

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Schémas Pydantic ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age en années")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Température en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Présence de toux")
    fatigue: bool = Field(..., description="Présence de fatigue")
    maux_tete: bool = Field(..., description="Présence de maux de tête")
    region: str = Field(..., description="Région du Sénégal")

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

# --- Chargement du modèle (une seule fois au démarrage) ---
print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modele charge : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")

# --- Routes ---
@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {"status": "ok", "message": "SenSante API is running"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """Prédire un diagnostic à partir des symptômes d'un patient."""

    # 1. Encoder les variables catégoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F.")

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}")

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Prédire
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    # 4. Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Message adapté
    messages = {
        "paludisme": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typhoide": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller."
    }

    # 6. Retourner le résultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )
@app.get("/model-info")
def model_info():
    """Informations sur le modèle chargé."""
    return {
        "type": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "nombre_features": model.n_features_in_
    }