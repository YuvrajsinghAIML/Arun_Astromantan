# ==========================================
# 1. LOAD DATA
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading data...")
# TODO: Change 'your_dataset.csv' to your actual file name
# df = pd.read_csv("your_dataset.csv") 

# For testing the template, we'll create a dummy dataframe. 
# DELETE these 3 lines during the hackathon and uncomment the line above!
dummy_data = {'Feature1': [10, 15, 8, 20, 5, 22], 'Feature2': [1, 2, 1, 3, 1, 3], 'Category': [0, 1, 0, 1, 0, 1]}
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
# (Optional) Uncomment the next lines to see how categories are distributed
# sns.countplot(x='Category', data=df)
# plt.title("Category Distribution")
# plt.show()

# ==========================================
# 5. SPLIT DATA
# ==========================================
# TODO: Put your feature columns inside the double brackets
X = df[["Feature1", "Feature2"]] 

# TODO: Put the CATEGORY column you are trying to predict here
y = df["Category"] 

# Splits 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 6. TRAIN MODEL
# ==========================================
print("Training Logistic Regression model...")
# Using Ridge (L2) as recommended in your PDF
model = LogisticRegression(penalty="l2", solver="liblinear", C=1)
model.fit(X_train, y_train)

# ==========================================
# 7. EVALUATE
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"Model Accuracy: {accuracy * 100:.1f}%")
print("-" * 30)

# Detailed report showing precision and recall
# print("Detailed Report:\n", classification_report(y_test, y_pred))
