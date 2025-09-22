import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load dataset
data = pd.read_csv("disease_prediction_sample.csv")

# Step 2: Drop PatientID (not useful for prediction)
data = data.drop("PatientID", axis=1)

# Step 3: Encode categorical values
encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(data["Gender"])   # M=1, F=0

for col in ["Fever","Cough","Headache","Sore Throat","Fatigue","Nausea"]:
    data[col] = data[col].map({"Yes":1, "No":0})

print("Cleaned Data:\n", data.head())

# Step 4: Split features and labels
X = data.drop("Diagnosis", axis=1)   # input features
y = data["Diagnosis"]                # target (disease)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Test accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")