import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict

class MineLSTMPredictor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        """
        LSTM model for predicting mine conditions.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of features in the hidden state
            num_layers: Number of recurrent layers
            output_size: Number of output features
            dropout: Dropout probability
        """
        super(MineLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Initial hidden state and cell state
            
        Returns:
            output: Predicted values
            (h_n, c_n): Final hidden state and cell state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # LSTM forward pass
        out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Get predictions from the last time step
        out = self.fc(out[:, -1, :])
        
        return out, (h_n, c_n)
    
    def predict_sequence(self, x: torch.Tensor, 
                        sequence_length: int = 10) -> np.ndarray:
        """
        Predict a sequence of values.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            sequence_length: Number of steps to predict ahead
            
        Returns:
            predictions: Array of predicted values
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            # Initial prediction
            pred, (h_n, c_n) = self(x)
            predictions.append(pred.cpu().numpy())
            
            # Generate subsequent predictions
            current_input = x
            for _ in range(sequence_length - 1):
                # Update input sequence
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    pred.unsqueeze(1)
                ], dim=1)
                
                # Make prediction
                pred, (h_n, c_n) = self(current_input, (h_n, c_n))
                predictions.append(pred.cpu().numpy())
        
        return np.array(predictions).squeeze()

    def explain_prediction(self, prediction: float) -> Dict:
        """Provide interpretable explanation for predictions."""
        return {
            'prediction': prediction,
            'confidence': self._calculate_confidence(),
            'contributing_factors': self._identify_factors(),
            'recommended_actions': self._suggest_actions()
        }

class MineAnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        """
        Anomaly detection model using Isolation Forest.
        
        Args:
            contamination: Expected proportion of anomalies in the data
        """
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
    
    def fit(self, X: np.ndarray) -> None:
        """Fit the anomaly detection model."""
        self.model.fit(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in the data."""
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for the data."""
        return self.model.score_samples(X)

class MineHazardClassifier:
    def __init__(self, n_estimators: int = 100):
        """
        Hazard classification model using Random Forest.
        
        Args:
            n_estimators: Number of trees in the forest
        """
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the classification model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hazard types."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates for each class."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_ 