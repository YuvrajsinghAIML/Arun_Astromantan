import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. LOAD DATA
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

# 2. TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================
# 3. SHOWCASE: GENERATE GRAPHS FOR JUDGES
# ==========================================
print("Generating graphs... (Close the first graph to see the second one)")

# GRAPH 1: Correlation Heatmap (Shows how features relate to temperature)
plt.figure(figsize=(10, 8))
# Combine features and target for the heatmap
heatmap_data = df[features + ['mean_temperature']]
sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: What affects Planetary Temperature?", fontsize=14)
plt.tight_layout()
plt.savefig("Showcase_Graph_1_Heatmap.png")
plt.show()

# GRAPH 2: Actual vs Predicted (Proves the model works)
plt.figure(figsize=(10, 6))
# Plot a perfect line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--', label='Perfect Prediction')
# Scatter plot of our predictions
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Our Model predictions')

plt.title("Model Accuracy: Actual vs Predicted Temperature", fontsize=14)
plt.xlabel("Actual Temperature (Kelvin)")
plt.ylabel("Predicted Temperature (Kelvin)")
plt.legend()
plt.tight_layout()
plt.savefig("Showcase_Graph_2_Accuracy.png")
plt.show()

print("Graphs saved to your folder as PNG images so you can put them in your presentation!")
