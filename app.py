from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load pre-trained model
model = xgb.Booster()
model.load_model('models/xgboost_fraud_model.json')

# Function to calculate risk score based on wallet address
def calculate_risk(wallet_address):
    # Here, you'll need to implement your logic to fetch or compute actual features
    # For demonstration, we're using dummy values; replace this with real data logic
    # You may need to fetch transaction data for the wallet address here

    # Dummy feature creation for demonstration (replace this with actual logic)
    features = pd.DataFrame({
        'wallet_age': [1],  # Placeholder: replace with actual calculation
        'transaction_frequency': [10],  # Placeholder: replace with actual calculation
        'balance_change': [100]  # Placeholder: replace with actual calculation
    })

    # Convert DataFrame to DMatrix for prediction
    dmatrix = xgb.DMatrix(features)
    
    # Predict risk score
    risk_score = model.predict(dmatrix)
    return risk_score[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    wallet_address = request.form['wallet_address']
    print(f"Received wallet address: {wallet_address}")  # Debugging line

    # Call the risk calculation function
    risk_score = calculate_risk(wallet_address)
    print(f"Calculated risk score: {risk_score}")  # Debugging line

    # Render the result page with the score
    return render_template('result.html', score=risk_score, wallet_address=wallet_address)

if __name__ == '__main__':
    app.run(debug=True)
