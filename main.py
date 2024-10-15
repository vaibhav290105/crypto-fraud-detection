'''
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Function to load and preprocess data
def load_data():
    # Load the dataset
    data_path = os.path.join('data', 'crypto_transactions.csv')
    df = pd.read_csv(data_path)

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
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize XGBoost classifier
    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    # Save the trained model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_model(os.path.join(model_dir, 'xgboost_fraud_model.json'))

    return model

if __name__ == '__main__':
    # Load data
    X, y = load_data()

    # Train model
    train_model(X, y)'''
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Function to load and preprocess data
def load_data():
    data_path = os.path.join('data', 'crypto_transactions.csv')
    df = pd.read_csv(data_path)

    # Feature engineering
    df['wallet_age'] = df['transaction_time'] - df['wallet_creation_time']
    df['transaction_frequency'] = df.groupby('wallet_address')['transaction_id'].transform('count')
    df['balance_change'] = df['transaction_value'] - df['previous_balance']

    X = df[['wallet_age', 'transaction_frequency', 'balance_change']]
    y = df['is_fraud']

    # Check class distribution
    print("Class distribution in the dataset:")
    print(y.value_counts())

    return X, y

# Function to train the model
def train_model(X, y):
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Train XGBoost model
    model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100)
    model.fit(X_resampled, y_resampled)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

    # Save the trained model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_model(os.path.join(model_dir, 'xgboost_fraud_model.json'))

    return model

if __name__ == '__main__':
    # Load data
    X, y = load_data()

    # Train model
    train_model(X, y)
