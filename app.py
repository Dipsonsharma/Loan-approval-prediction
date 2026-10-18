import os
import pickle

import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and preprocessing object once when the app starts.
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept either form data (HTML form) or JSON data (API request).
        data = request.get_json(silent=True) if request.is_json else request.form

        features = pd.DataFrame([{
            "no_of_dependents": int(data["no_of_dependents"]),
            "education": data["education"],
            "self_employed": data["self_employed"],
            "income_annum": float(data["income_annum"]),
            "loan_amount": float(data["loan_amount"]),
            "loan_term": float(data["loan_term"]),
            "cibil_score": float(data["cibil_score"]),
            "Total_asset_value": float(data["Total_asset_value"]),
        }])

        # Apply the same fitted preprocessing used during training.
        features_processed = preprocessor.transform(features)

        prediction = int(model.predict(features_processed)[0])
        loan_status = "Approved" if prediction == 0 else "Rejected"

        # JSON is convenient for a separate frontend that uses fetch().
        if request.is_json:
            result = {"prediction": loan_status}

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features_processed)[0]
                class_index = list(model.classes_).index(prediction)
                result["confidence"] = float(probabilities[class_index])

            return jsonify(result)

        # For a regular HTML form submission, display the result on index.html.
        return render_template("index.html", prediction_text=f"Loan Status: {loan_status}")

    except KeyError as error:
        message = f"Missing input field: {error.args[0]}"
        if request.is_json:
            return jsonify({"error": message}), 400
        return render_template("index.html", error=message), 400

    except (TypeError, ValueError) as error:
        message = f"Invalid input: {error}"
        if request.is_json:
            return jsonify({"error": message}), 400
        return render_template("index.html", error=message), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)
