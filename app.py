from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    age = int(request.form["age"])
    gender = int(request.form["gender"])  # 0 = Female, 1 = Male
    fever = int(request.form["fever"])
    cough = int(request.form["cough"])
    headache = int(request.form["headache"])
    sore_throat = int(request.form["sore_throat"])
    fatigue = int(request.form["fatigue"])
    nausea = int(request.form["nausea"])

    # Create feature array
    features = np.array([[age, gender, fever, cough, headache, sore_throat, fatigue, nausea]])
    prediction = model.predict(features)[0]

    return render_template("index.html", result=f"Predicted Disease: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)