import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir features (X) et cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")
print(f"Cible : {y.shape}")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")
from sklearn.ensemble import RandomForestClassifier

# Créer et entraîner le modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
  
# Prédire sur les données de test
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
import joblib
import os

# Créer le dossier models/ si nécessaire
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")

# Sauvegarder les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size/1024:.1f} Ko")
print("Encodeurs et metadata sauvegardes.")
# Recharger le modèle depuis le fichier
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modele recharge : RandomForestClassifier")

# Nouveau patient test
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'], sexe_enc,
    nouveau_patient['temperature'], nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']), int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']), region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]

print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {probas.max():.1%}")
# Exercice 1 : Importance des features
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print(f"{name:20s} : {imp:.3f}")
# Exercice 2 : 3 patients fictifs
patients = [
    {'age': 10, 'sexe': 'M', 'temperature': 37.0,
     'tension_sys': 110, 'toux': False, 'fatigue': False,
     'maux_tete': False, 'region': 'Dakar'},
    {'age': 35, 'sexe': 'F', 'temperature': 40.5,
     'tension_sys': 130, 'toux': True, 'fatigue': True,
     'maux_tete': True, 'region': 'Dakar'},
    {'age': 70, 'sexe': 'M', 'temperature': 38.5,
     'tension_sys': 150, 'toux': True, 'fatigue': True,
     'maux_tete': False, 'region': 'Dakar'},
]

for i, p in enumerate(patients):
    s = le_sexe_loaded.transform([p['sexe']])[0]
    r = le_region_loaded.transform([p['region']])[0]
    f = [p['age'], s, p['temperature'], p['tension_sys'],
         int(p['toux']), int(p['fatigue']), int(p['maux_tete']), r]
    diag = model_loaded.predict([f])[0]
    proba = model_loaded.predict_proba([f])[0].max()
    print(f"Patient {i+1} ({p['age']}ans, {p['sexe']}) : {diag} ({proba:.1%})")
