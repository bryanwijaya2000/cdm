# Import required libraries
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from fastapi import FastAPI
from json import loads

# Initialize the API
app = FastAPI()

# List variable to store the food names
food_names = [
    "alfredo",
    "broccoli",
    "brownie",
    "cake",
    "carrot",
    "cereal",
    "cheese",
    "chicken",
    "chocolate",
    "coffee",
    "cookie",
    "corn",
    "couscous",
    "crab",
    "donut",
    "egg",
    "fajitas",
    "fries",
    "grilledcheese",
    "hotdog",
    "icecream",
    "macncheese",
    "nachos",
    "nuggets",
    "rice",
    "salad",
    "salmon",
    "shrimp",
    "soup",
    "steak",
    "sushi",
    "tartare"
]

# Variable to store the food nutrition dataset
food_nutrition_data = pd.read_csv("food_nutrition.csv")

# Health Data/Record Format: [ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]
# ap_hi: Systolic blood pressure.
# ap_lo: Diastolic blood pressure.
# cholesterol: Cholesterol levels. Categorical variable (1: Normal, 2: Above Normal, 3: Well Above Normal).
# gluc: Glucose levels. Categorical variable (1: Normal, 2: Above Normal, 3: Well Above Normal).
# smoke: Smoking status. Binary variable (0: Non-smoker, 1: Smoker).
# alco: Alcohol intake. Binary variable (0: Does not consume alcohol, 1: Consumes alcohol).
# active: Physical activity. Binary variable (0: Not physically active, 1: Physically active).

# List variable to store the health data (dummy)
health_data = np.array([
    [110, 80, 1, 1, 0, 0, 1],
    [140, 90, 3, 1, 0, 0, 1],
    [130, 70, 3, 1, 0, 0, 0]
])

# Endpoint to get the category of the blood pressure reading according to the given health record represented as a string which values are separated by a single comma
# Input format: ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active
@app.get("/get-blood-pressure-reading-category")
async def GetBloodPressureReadingCategory(health_record_str: str):
    health_record = list(map(float, health_record_str.split(",")))
    health_record_json = {}
    fields = ["systolic", "diastolic", "cholestrol_level_cat", "glucose_level_cat", "smokes", "drinks_alcohol", "is_physically_active"]
    cat3classes = ["Normal", "Above Normal", "Well Above Normal"]
    cat2classes = ["No", "Yes"]
    for i in range(len(fields)):
        if fields[i] == "cholestrol_level_cat" or fields[i] == "glucose_level_cat":
            health_record_json[fields[i]] = cat3classes[int(health_record[i]) - 1]
        elif fields[i] == "smokes" or fields[i] == "drinks_alcohol" or fields[i] == "is_physically_active":
            health_record_json[fields[i]] = cat2classes[int(health_record[i])]
        else:
            health_record_json[fields[i]] = health_record[i]
    systolic = health_record[0]
    diastolic = health_record[1]
    bpReadingCat = "Unknown"
    if systolic < 120 and diastolic < 80:
        bpReadingCat = "Normal"
    elif (systolic >= 120 and systolic <= 129) and diastolic < 80:
        bpReadingCat = "Elevated"
    elif (systolic >= 130 and systolic <= 139) or (diastolic >= 80 and diastolic <= 89):
        bpReadingCat = "Hypertension Stage 1"
    elif systolic >= 140 or diastolic >= 90:
        bpReadingCat = "Hypertension Stage 2"
    elif systolic > 180 or diastolic > 120:
        bpReadingCat = "Hypertensive Crisis"
    return {
        "Health-Record": health_record_json,
        "BP-Reading-Cat": bpReadingCat
    }

# Endpoint to display the risk score according to the given health record using the trained Machine Learning model
# The risk score ranges from 0 to 1 inclusive
# Displays an array representing the risk probability of each blood pressure reading category as returned by the function GetBloodPressureReadingCategory
@app.get("/show-risk-score")
async def ShowRiskScore():
    global health_data
    num_records = health_data.shape[0]
    latest_health_record = health_data[num_records - 1]
    latest_health_record_json = {}
    fields = ["systolic", "diastolic", "cholestrol_level_cat", "glucose_level_cat", "smokes", "drinks_alcohol", "is_physically_active"]
    cat3classes = ["Normal", "Above Normal", "Well Above Normal"]
    cat2classes = ["No", "Yes"]
    for i in range(len(fields)):
        if fields[i] == "cholestrol_level_cat" or fields[i] == "glucose_level_cat":
            latest_health_record_json[fields[i]] = cat3classes[int(latest_health_record[i]) - 1]
        elif fields[i] == "smokes" or fields[i] == "drinks_alcohol" or fields[i] == "is_physically_active":
            latest_health_record_json[fields[i]] = cat2classes[int(latest_health_record[i])]
        else:
            latest_health_record_json[fields[i]] = float(latest_health_record[i])
    risk_scores = mlModel.predict_proba([latest_health_record])[0]
    bpReadingCats = ["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis"]
    risk_scores_json = {}
    for i in range(len(bpReadingCats)):
        if i < len(risk_scores):
            risk_scores_json[bpReadingCats[i]] = str(int(risk_scores[i] * 100)) + "%"
        else:
            risk_scores_json[bpReadingCats[i]] = "0%"
    return {
        "Latest-Health-Record": latest_health_record_json,
        "Risk-Scores": risk_scores_json
    }

# Endpoint to get the food data according to the given food image file path
# Output format: [Food Name,Calcium,Calories,Carbs,Cholesterol,Copper,Fats,Fiber,Folate,Iron,Magnesium,Monounsaturated Fat,Net carbs,Omega-3 - DHA,Omega-3 - DPA,Omega-3 - EPA,Phosphorus,Polyunsaturated fat,Potassium,Protein,Saturated Fat,Selenium,Sodium,Trans Fat,Vitamin A (IU),Vitamin A RAE,Vitamin B1,Vitamin B12,Vitamin B2,Vitamin B3,Vitamin B5,Vitamin B6,Vitamin C,Zinc,Choline,Fructose,Histidine,Isoleucine,Leucine,Lysine,Manganese,Methionine,Phenylalanine,Starch,Sugar,Threonine,Tryptophan,Valine,Vitamin D,Vitamin E,Vitamin K,Omega-3 - ALA,Omega-6 - Eicosadienoic acid,Omega-6 - Gamma-linoleic acid,Omega-3 - Eicosatrienoic acid,Omega-6 - Dihomo-gamma-linoleic acid,Omega-6 - Linoleic acid,Omega-6 - Arachidonic acid]
@app.get("/get-food-data")
async def GetFoodData(food_image_dir_path: str):
    global food_names, food_nutrition_data
    foodData = None
    food_image = cv2.imread(food_image_dir_path)
    food_image = cv2.resize(food_image, (200, 200))
    food_image = cv2.cvtColor(food_image, cv2.COLOR_BGR2RGB)
    food_image = np.array(food_image)
    food_image_arr = np.array([food_image])
    pred = np.argmax(cvModel.predict(food_image_arr))
    food_name = food_names[pred]
    for _, row in food_nutrition_data.iterrows():
        if row["Food Name"] in food_name or food_name in row["Food Name"]:
            foodData = row.to_frame().T
            foodData = foodData.drop("Category Name", axis=1)
            break
    return {
        "Food-Image_Dir_Path": food_image_dir_path,
        "Food-Data": loads(foodData.to_json(orient="records"))[0]
    }

# Machine Learning (ML)

# Load the trained Machine Learning model
with open("ml_model.pkl", "rb") as f:
    mlModel = pickle.load(f)

# Image Recognition

# Load the trained Image Recognition model
cvModel = load_model("cv_model.keras")