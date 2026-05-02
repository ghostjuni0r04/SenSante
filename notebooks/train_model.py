import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# ============================================================
# ETAPE 1 : Charger le dataset
# ============================================================
df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# ============================================================
# ETAPE 2 : Preparer les features
# ============================================================
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")
print(f"Cible : {y.shape}")

# ============================================================
# ETAPE 3 : Separer entrainement et test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEntrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# ============================================================
# ETAPE 4 : Entrainer le modele
# ============================================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Classes : {list(model.classes_)}")

# ============================================================
# ETAPE 5 : Evaluer le modele
# ============================================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# ============================================================
# ETAPE 6 : Serialiser le modele
# ============================================================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"\nModele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")
print("Encodeurs et metadata sauvegardes.")

# ============================================================
# ETAPE 7 : Tester le modele serialise
# ============================================================
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'frissons': True,
    'nausee': False,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    int(nouveau_patient['frissons']),
    int(nouveau_patient['nausee']),
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")

print("\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:10s} : {proba:.1%} {bar}")


# ============================================================
# EXERCICE 1 : Importance des features
# ============================================================
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")

# ============================================================
# EXERCICE 2 : Tester avec 3 patients fictifs
# ============================================================
print("\nRegions connues :", list(le_region_loaded.classes_))

patients_test = [
    {'age': 10, 'sexe': 'M', 'temperature': 37.0, 'tension_sys': 120,
     'toux': False, 'fatigue': False, 'maux_tete': False, 'frissons': False, 'nausee': False, 'region': le_region_loaded.classes_[0]},
    {'age': 35, 'sexe': 'F', 'temperature': 40.5, 'tension_sys': 130,
     'toux': True, 'fatigue': True, 'maux_tete': True, 'frissons': True, 'nausee': True, 'region': le_region_loaded.classes_[1]},
    {'age': 70, 'sexe': 'M', 'temperature': 38.5, 'tension_sys': 150,
     'toux': True, 'fatigue': True, 'maux_tete': False, 'frissons': False, 'nausee': False, 'region': le_region_loaded.classes_[2]},
]

print("\n--- Exercice 2 : 3 patients fictifs ---")
for i, p in enumerate(patients_test):
    sexe_enc = le_sexe_loaded.transform([p['sexe']])[0]
    region_enc = le_region_loaded.transform([p['region']])[0]
    features = [p['age'], sexe_enc, p['temperature'], p['tension_sys'],
                int(p['toux']), int(p['fatigue']), int(p['maux_tete']),
                int(p['frissons']), int(p['nausee']), region_enc]
    diag = model_loaded.predict([features])[0]
    proba = model_loaded.predict_proba([features])[0].max()
    print(f"Patient {i+1} ({p['sexe']}, {p['age']} ans, T°{p['temperature']}, {p['region']}) : {diag} ({proba:.1%})")