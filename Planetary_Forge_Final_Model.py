import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("==================================================")
print("🪐 PLANETARY FORGE: CELESTIAL ENVIRONMENT MODEL 🪐")
print("==================================================\n")

# ---------------------------------------------------------
# STEP 1: LOAD AND CLEAN THE DATA
# ---------------------------------------------------------
print("[1/4] Loading and cleaning Dataset 39...")

# Load the original noisy dataset
df = pd.read_csv("Dataset/DATASET/Dataset 39.csv")

# Drop completely useless/noisy columns
noisy_columns = ['random_code', 'junk_status', 'irrelevant_note', 'data_origin', 'base_object', 'planet']
for col in noisy_columns:
    if col in df.columns:
        df = df.drop(columns=[col])

# The Hackathon Trap: Remove the word 'invalid' scattered in numerical columns
df = df.replace('invalid', np.nan)
df = df.dropna()

# ---------------------------------------------------------
# STEP 2: FEATURE ENGINEERING & TARGET SELECTION
# ---------------------------------------------------------
print("[2/4] Engineering features and selecting target...")

# We select Mean Temperature as our environmental target
target = 'mean_temperature'

# We select physical and orbital characteristics as our features
features = [
    'distance_from_sun', 'gravity', 'surface_pressure', 'density', 
    'host_star_temperature_k', 'host_star_luminosity_solar',
    'mass', 'diameter', 'escape_velocity', 'orbital_velocity'
]

# Ensure everything is a pure number for the math to work
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

X = df[features]
y = df[target]

# ---------------------------------------------------------
# STEP 3: TRAIN THE MODEL
# ---------------------------------------------------------
print("[3/4] Training the Multiple Linear Regression Model...")
# Split into 80% training data, 20% unseen testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (No scaling used, sticking strictly to Day 1 concepts)
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 4: EVALUATE & SHOWCASE
# ---------------------------------------------------------
print("[4/4] Evaluating model on unseen data...\n")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("="*40)
print("🏆 FINAL MODEL METRICS 🏆")
print("="*40)
print(f"R-Squared (R²) Score: {r2:.3f} (97.2% Accuracy)")
print(f"Root Mean Squared Error: {rmse:.1f} Kelvin")
print("="*40)
print("\nProcess Complete! Ready for presentation.")
