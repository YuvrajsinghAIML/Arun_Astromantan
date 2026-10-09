import pandas as pd
import numpy as np

# 1. LOAD THE DATA
print("--- STEP 1: LOADING DATA ---")
df = pd.read_csv("Dataset/DATASET/Dataset 39.csv")
print(f"Original Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nAll columns in Dataset 39:")
print(list(df.columns))

# 2. DETECT NOISY/REDUNDANT COLUMNS
print("\n--- STEP 2: DETECTING NOISY DATA ---")
# Let's look at the columns that seem irrelevant to physics
noisy_columns = ['random_code', 'junk_status', 'irrelevant_note', 'data_origin', 'base_object']

for col in noisy_columns:
    if col in df.columns:
        print(f"Found noisy column: {col} -> Dropping it.")
        df = df.drop(columns=[col])

# Let's also drop the 'planet' name column as it's an identifier, not a physical feature
if 'planet' in df.columns:
    df = df.drop(columns=['planet'])

print(f"\nData Shape after dropping noise: {df.shape[0]} rows, {df.shape[1]} columns")

# 3. CHECK FOR MISSING VALUES
print("\n--- STEP 3: MISSING VALUES ---")
missing_data = df.isna().sum()
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    print("Columns with missing data:")
    print(missing_data)
    
    # We will drop rows with missing values for now (as taught in the Astromanthan PDF)
    df_clean = df.dropna()
    print(f"\nData Shape after dropping missing values: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
else:
    print("No missing values found!")
    df_clean = df

# 4. IDENTIFY OUR TARGET VARIABLE
print("\n--- STEP 4: IDENTIFYING THE TARGET ---")
print("The Problem Statement asks us to 'Develop a predictive model for a relevant planetary property.'")
print("Looking at the columns, predicting 'mean_temperature' (environmental condition) is the perfect target!")

# Save the clean dataset for the next step
df_clean.to_csv("cleaned_planetary_data.csv", index=False)
print("\nSuccess! Cleaned data saved as 'cleaned_planetary_data.csv'.")
