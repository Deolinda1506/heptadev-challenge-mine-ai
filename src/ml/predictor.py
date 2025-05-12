import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class MinePredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lstm_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_sequence_data(self, df: pd.DataFrame, 
                            sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM model."""
        data = df['value'].values
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def train_lstm(self, df: pd.DataFrame, 
                  input_size: int = 1,
                  hidden_size: int = 64,
                  num_layers: int = 2,
                  epochs: int = 100):
        """Train LSTM model for time series prediction."""
        X, y = self.prepare_sequence_data(df)
        X = X.reshape(-1, X.shape[1], input_size)
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X.reshape(-1, input_size)).reshape(X.shape)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).ravel()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # Initialize model
        self.lstm_model = LSTMPredictor(input_size, hidden_size, num_layers, 1)
        self.lstm_model.to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        
        # Training loop
        for epoch in range(epochs):
            self.lstm_model.train()
            optimizer.zero_grad()
            
            outputs = self.lstm_model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict_next_values(self, df: pd.DataFrame, 
                          steps_ahead: int = 5) -> np.ndarray:
        """Predict future values using LSTM model."""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained. Call train_lstm first.")
            
        self.lstm_model.eval()
        with torch.no_grad():
            last_sequence = df['value'].values[-10:].reshape(1, -1, 1)
            last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, -1, 1)
            
            predictions = []
            current_sequence = torch.FloatTensor(last_sequence_scaled).to(self.device)
            
            for _ in range(steps_ahead):
                pred = self.lstm_model(current_sequence)
                predictions.append(pred.item())
                
                # Update sequence for next prediction
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.reshape(1, 1, 1)
                ], dim=1)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.ravel()
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        features = df[['value']].values
        self.anomaly_detector.fit(features)
        
        df['anomaly_score'] = self.anomaly_detector.score_samples(features)
        df['is_anomaly'] = self.anomaly_detector.predict(features) == -1
        
        return df
    
    def classify_hazard_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify the type of hazard based on sensor data patterns."""
        # This is a simplified version - in reality, you'd need more sophisticated features
        features = df[['value', 'anomaly_score']].values
        df['hazard_type'] = self.classifier.predict(features)
        
        return df
    
    def save_models(self):
        """Save trained models to disk."""
        if self.lstm_model is not None:
            torch.save(self.lstm_model.state_dict(), 
                      os.path.join(self.model_dir, 'lstm_model.pth'))
        
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump(self.anomaly_detector, 
                   os.path.join(self.model_dir, 'anomaly_detector.pkl'))
        joblib.dump(self.classifier, 
                   os.path.join(self.model_dir, 'classifier.pkl'))
    
    def load_models(self):
        """Load trained models from disk."""
        lstm_path = os.path.join(self.model_dir, 'lstm_model.pth')
        if os.path.exists(lstm_path):
            self.lstm_model = LSTMPredictor(1, 64, 2, 1)
            self.lstm_model.load_state_dict(torch.load(lstm_path))
            self.lstm_model.to(self.device)
        
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.anomaly_detector = joblib.load(
            os.path.join(self.model_dir, 'anomaly_detector.pkl'))
        self.classifier = joblib.load(
            os.path.join(self.model_dir, 'classifier.pkl'))

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    values = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    df = pd.DataFrame({'timestamp': dates, 'value': values})
    
    # Initialize and train predictor
    predictor = MinePredictor()
    predictor.train_lstm(df)
    
    # Make predictions
    future_values = predictor.predict_next_values(df)
    print("Predicted future values:", future_values)
    
    # Detect anomalies
    df_with_anomalies = predictor.detect_anomalies(df)
    print("\nDetected anomalies:")
    print(df_with_anomalies[df_with_anomalies['is_anomaly']]) 