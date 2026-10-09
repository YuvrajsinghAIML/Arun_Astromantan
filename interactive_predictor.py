import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Quickly train the model quietly in the background
df = pd.read_csv("cleaned_planetary_data.csv")
df = df.replace('invalid', np.nan).dropna()

features = [
    'distance_from_sun', 'gravity', 'surface_pressure', 'density', 
    'host_star_temperature_k', 'host_star_luminosity_solar',
    'mass', 'diameter', 'escape_velocity', 'orbital_velocity'
]

for col in features + ['mean_temperature']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

X = df[features]
y = df['mean_temperature']

model = LinearRegression()
model.fit(X, y) # Training on all clean data for maximum accuracy during the live demo

print("\n=======================================================")
print("🚀 WELCOME TO THE PLANETARY TEMPERATURE PREDICTOR 🚀")
print("=======================================================\n")
print("Judges, please enter the physical characteristics of your custom planet:")

try:
    # 2. Ask the judge for their inputs
    dist = float(input("1. Distance from Sun (e.g., 1.0 for Earth): "))
    grav = float(input("2. Gravity (e.g., 9.8 for Earth): "))
    pres = float(input("3. Surface Pressure (e.g., 1.0 for Earth): "))
    dens = float(input("4. Density (e.g., 5.5 for Earth): "))
    star_temp = float(input("5. Host Star Temperature (e.g., 5778 for our Sun): "))
    star_lum = float(input("6. Host Star Luminosity (e.g., 1.0 for our Sun): "))
    mass = float(input("7. Planet Mass (e.g., 1.0 for Earth): "))
    diam = float(input("8. Planet Diameter (e.g., 1.0 for Earth): "))
    esc_vel = float(input("9. Escape Velocity (e.g., 11.2 for Earth): "))
    orb_vel = float(input("10. Orbital Velocity (e.g., 29.7 for Earth): "))
    
    # 3. Format the input for the model
    custom_planet = pd.DataFrame([[
        dist, grav, pres, dens, star_temp, star_lum, mass, diam, esc_vel, orb_vel
    ]], columns=features)
    
    # 4. Make the prediction
    predicted_temp = model.predict(custom_planet)[0]
    
    print("\n" + "="*50)
    print("✨ CALCULATING CELESTIAL ENVIRONMENT... ✨")
    print("="*50)
    print(f"Based on the physics provided, the predicted Mean Temperature is:")
    print(f">>> {predicted_temp:.2f} Kelvin <<<")
    print("="*50 + "\n")

except ValueError:
    print("\nError: Please enter numbers only! Restart the script to try again.")
