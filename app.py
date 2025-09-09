# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the model
# try:
#     with open(r'D:\Appro_Project_1(image_classification)\prediction\copy-heart\model\heart.pickle', 'rb') as file:
#         model1 = pickle.load(file)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model1 = None

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         d1 = int(request.form['age'])
#         d2 = int(request.form['sex'])        
#         d3 = int(request.form['cp'])
#         d4 = int(request.form['trestbps'])
#         d5 = int(request.form['chol'])
#         d6 = int(request.form['fbs'])
#         d7 = int(request.form['restecg'])
#         d8 = int(request.form['thalach'])
#         d9 = int(request.form['exang'])
#         d10 = float(request.form['oldpeak'])  # should be float
#         d11 = int(request.form['slope'])  
#         d12 = int(request.form['ca'])
#         d13 = int(request.form['thal'])

#         # Create input array for prediction
#         arr = np.array([[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13]])

#         # Predict
#         if model1:
#             pred1 = model1.predict(arr)
#             risk = int(pred1[0])
#         else:
#             risk = 0  # fallback if model isn't loaded

#         # Pass result to template
#         return render_template("result.html", risk=risk)

#     except Exception as e:
#         return f"An error occurred: {e}"

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np
import sys, traceback

app = Flask(__name__)

# -------------------------------
# Load the model
# -------------------------------
try:
    with open(r"D:\Appro_Project_1(image_classification)\heart-disease\model\heart (1).pickle", "rb") as file:
        model1 = pickle.load(file)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Loading model failed: {e}")
    model1 = None

# Feature order must match training dataset!
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("\n====== /predict CALLED ======")
        print("[RAW FORM] ->", request.form.to_dict())

        # Parse inputs safely
        d1 = int(request.form.get("age", 0))
        d2 = int(request.form.get("sex", 0))
        d3 = int(request.form.get("cp", 0))
        d4 = int(request.form.get("trestbps", 0))
        d5 = int(request.form.get("chol", 0))
        d6 = int(request.form.get("fbs", 0))
        d7 = int(request.form.get("restecg", 0))
        d8 = int(request.form.get("thalach", 0))
        d9 = int(request.form.get("exang", 0))
        d10 = float(request.form.get("oldpeak", 0))
        d11 = int(request.form.get("slope", 0))
        d12 = int(request.form.get("ca", 0))
        d13 = int(request.form.get("thal", 0))

        # Create array in correct order
        arr = np.array([[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13]], dtype=float)

        print("[ORDER ] ->", FEATURES)
        print("[PARSED] ->", arr.tolist())
        print("[SHAPE ] ->", arr.shape)

        # -------------------------------
        # Prediction
        # -------------------------------
        if model1:
            # Check if model supports probability
            proba, score = None, None
            if hasattr(model1, "predict_proba"):
                proba = model1.predict_proba(arr)[0]
                print("[PROBA ] ->", proba)
            if hasattr(model1, "decision_function"):
                score = model1.decision_function(arr)
                print("[DECISION] ->", score)

            pred1 = model1.predict(arr)
            risk = int(pred1[0])
            print("[PRED  ] ->", risk)
        else:
            print("[WARN] Model not loaded, defaulting risk=0")
            risk = 0

        # -------------------------------
        # Pass result to template
        # -------------------------------
        return render_template(
            "result.html",
            risk=risk,
            features=FEATURES,
            values=arr.tolist()[0],
            proba=None if proba is None else [float(p) for p in proba],
        )

    except Exception as e:
        print("[EXCEPTION]", e, file=sys.stderr)
        traceback.print_exc()
        return f"An error occurred: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)

