import pickle
import os
import numpy as np
from flask import Flask, render_template, request

# Get the directory where this script is located
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model.pkl")

app = Flask(__name__, template_folder="templates")

# Load regression model and feature list saved by the training script
with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)

# `saved` is a dict with keys: 'model' and 'features'
model = saved['model']
FEATURE_NAMES = saved['features']
# boolean features in the saved feature list (use checkboxes in the UI)
BOOL_FEATURES = [f for f in FEATURE_NAMES if f.lower() in ('tutoring','extracurricular','sports','music','volunteering')]

# Precompute feature importances if available
FEATURE_IMPORTANCES = None
if hasattr(model, 'feature_importances_'):
    try:
        imp = list(zip(FEATURE_NAMES, model.feature_importances_))
        # sort descending
        imp_sorted = sorted(imp, key=lambda x: x[1], reverse=True)
        FEATURE_IMPORTANCES = imp_sorted
    except Exception:
        FEATURE_IMPORTANCES = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Build input vector reading fields by FEATURE_NAMES
        vals = []
        for name in FEATURE_NAMES:
            # handle boolean checkbox features
            if name in BOOL_FEATURES:
                # checkbox: present in form => on
                raw = request.form.get(name)
                val = 1.0 if raw in ('on', '1', 'true', 'True') else 0.0
            else:
                raw = request.form.get(name)
                if raw is None or raw == "":
                    val = 0.0
                else:
                    try:
                        val = float(raw)
                    except Exception:
                        val = 0.0
            vals.append(val)

        user_input = np.array([vals])

        print(f"Input features ({FEATURE_NAMES}): {user_input}")

        # Make prediction (regression -> numeric GPA)
        prediction = float(model.predict(user_input)[0])

        # estimate uncertainty from ensemble (std of tree predictions) when available
        uncertainty = None
        try:
            if hasattr(model, 'estimators_'):
                tree_preds = np.array([est.predict(user_input)[0] for est in model.estimators_])
                uncertainty = float(tree_preds.std())
        except Exception:
            uncertainty = None

        # pass feature importances to template
        return render_template("results.html", prediction=round(prediction,3), uncertainty=uncertainty, importances=FEATURE_IMPORTANCES)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("error.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
