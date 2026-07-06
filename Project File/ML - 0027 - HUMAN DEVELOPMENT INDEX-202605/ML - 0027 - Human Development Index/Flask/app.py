from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open('HDI.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Model not loaded. Please train the model first.", category=None)
    
    try:
        # Extract features from form
        life_exp = float(request.form['life_expectancy'])
        mean_school = float(request.form['mean_schooling'])
        exp_school = float(request.form['expected_schooling'])
        gni = float(request.form['gni_per_capita'])
        
        # Prepare feature array
        features = np.array([[life_exp, mean_school, exp_school, gni]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Cap the prediction to realistic HDI limits
        prediction = max(0.0, min(1.0, prediction))
        
        # Categorize
        if prediction >= 0.800:
            category = "Very High"
        elif prediction >= 0.700:
            category = "High"
        elif prediction >= 0.550:
            category = "Medium"
        else:
            category = "Low"
            
        prediction_text = f"{prediction:.3f}"
        
        return render_template('index.html', prediction=prediction_text, category=category,
                               life_exp=life_exp, mean_school=mean_school, exp_school=exp_school, gni=gni)
    except Exception as e:
        return render_template('index.html', prediction=f"Error processing input: {e}", category=None)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
