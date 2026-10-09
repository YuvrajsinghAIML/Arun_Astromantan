import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD CLEANED DATA
print("Loading cleaned data...")
df = pd.read_csv("cleaned_planetary_data.csv")

# 1.5 SNEAKY HACKATHON TRICK REMOVAL
print("Cleaning out 'invalid' strings scattered in the data...")
df = df.replace('invalid', np.nan)
df = df.dropna()

# 2. FEATURE ENGINEERING (Adding MORE features to improve Linear Regression)
features = [
    'distance_from_sun', 
    'gravity', 
    'surface_pressure', 
    'density', 
    'host_star_temperature_k', 
    'host_star_luminosity_solar',
    'mass',
    'diameter',
    'escape_velocity',
    'orbital_velocity'
]

# Convert columns to numbers (floats)
for col in features + ['mean_temperature']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

X = df[features]
y = df['mean_temperature']

# 3. SPLIT THE DATA
print(f"Splitting {len(df)} remaining pure rows into Training and Testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL
print("Training the Optimized Linear Regression Model (No Scaling)...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. EVALUATE THE MODEL
print("Evaluating the Model on unseen test data...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*40)
print("--- OPTIMIZED MODEL EVALUATION RESULTS ---")
print("="*40)
print(f"R-Squared (R²) Score: {r2:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.1f} Kelvin")
print("="*40)
