# ==========================================
# 1. LOAD DATA
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Loading data...")
# TODO: Change 'your_dataset.csv' to your actual file name
# df = pd.read_csv("your_dataset.csv") 

# For testing the template, we'll create a dummy dataframe. 
# DELETE these 3 lines during the hackathon and uncomment the line above!
dummy_data = {'Feature1': [1, 2, 3, 4, 5], 'Feature2': [5, 4, 3, 2, 1], 'TargetNumber': [10, 20, 30, 40, 50]}
df = pd.DataFrame(dummy_data)

# ==========================================
# 2. INSPECT & 3. CLEAN
# ==========================================
print("Data Shape before cleaning:", df.shape)
df = df.dropna() # Drops any rows with missing data
print("Data Shape after cleaning:", df.shape)

# ==========================================
# 4. EXPLORE
# ==========================================
# (Optional) Uncomment the next lines to see a heatmap of correlations
# sns.heatmap(df.corr(), annot=True)
# plt.title("Correlation Heatmap")
# plt.show()

# ==========================================
# 5. SPLIT DATA
# ==========================================
# TODO: Put your feature columns inside the double brackets
X = df[["Feature1", "Feature2"]] 

# TODO: Put the column you are trying to predict here
y = df["TargetNumber"] 

# Splits 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 6. TRAIN MODEL
# ==========================================
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 7. EVALUATE
# ==========================================
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("-" * 30)
print(f"Model R-Squared Score: {r2:.3f} (Closer to 1.0 is better!)")
print(f"Mean Squared Error: {mse:.3f} (Lower is better!)")
print("-" * 30)
