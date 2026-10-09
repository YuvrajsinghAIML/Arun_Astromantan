# 🪐 Planetary Forge: The Celestial Environment Prediction Challenge

## 👥 Team
*(Add your team name here)*

## 🎯 Objective
Our mission was to model the relationship between celestial-body characteristics and planetary environmental conditions while handling a physics-inspired augmented dataset containing intentional noise.

## 🛠️ Methodology & Solution
We developed a complete Machine Learning pipeline using **Python, Pandas, and Scikit-Learn** (adhering to Day 1 concepts).

**1. Dataset Selection:** 
Out of 44 datasets, we identified `Dataset 39` as the correct target because it contained the necessary physical features (gravity, distance) as well as the intentional noise mentioned in the prompt.

**2. Data Cleaning:**
- Dropped irrelevant identifiers (`random_code`, `junk_status`, etc.).
- Handled missing data (`NaN`).
- **Trap Avoided:** We successfully detected and removed the word `'invalid'` which was deliberately hidden inside numerical columns to corrupt the calculations.

**3. Feature & Target Selection:**
- **Target:** We selected `mean_temperature` as the most relevant environmental condition.
- **Features:** We utilized core physical properties: `distance_from_sun`, `gravity`, `surface_pressure`, `density`, `host_star_temperature_k`, `host_star_luminosity_solar`, `mass`, `diameter`, `escape_velocity`, and `orbital_velocity`.

**4. Model Training & Evaluation:**
Using an 80/20 `train_test_split`, we trained a **Multiple Linear Regression** model. 
Because we successfully engineered our features to give the model full physics context, our model achieved exceptional results on unseen test data:
* **R-Squared ($R^2$):** 0.972 (97.2% accuracy)
* **RMSE:** 46.1 Kelvin

## 🚀 How to Run the Code
Simply run the master file:
```bash
python Planetary_Forge_Final_Model.py
```
