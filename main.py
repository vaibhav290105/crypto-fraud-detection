import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load and preprocess data
def load_data():
    # Load your dataset (make sure to have this file in the data directory)
    df = pd.read_csv('data/crypto_transactions.csv')

    # Feature engineering
    df['wallet_age'] = df['transaction_time'] - df['wallet_creation_time']
    df['transaction_frequency'] = df.groupby('wallet_address')['transaction_id'].transform('count')
    df['balance_change'] = df['transaction_value'] - df['previous_balance']

    # Define features and target
    X = df[['wallet_age', 'transaction_frequency', 'balance_change']]
    y = df['is_fraud']

    return X, y

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')
    
    # Save the trained model
    model.save_model('models/xgboost_fraud_model.json')
    
    return model

if __name__ == '__main__':
    # Load data
    X, y = load_data()

    # Train model
    train_model(X, y)
