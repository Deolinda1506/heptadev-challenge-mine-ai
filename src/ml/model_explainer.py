import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self):
        self.feature_names = None
        self.model = None
        self.model_type = None
        
    def set_model(self, model, model_type: str, feature_names: List[str]):
        """Set the model and its type for explanation."""
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        
    def explain_prediction(self, input_data: np.ndarray) -> Dict:
        """Generate explanation for a single prediction."""
        try:
            if self.model_type == 'lstm':
                return self._explain_lstm_prediction(input_data)
            elif self.model_type == 'random_forest':
                return self._explain_rf_prediction(input_data)
            elif self.model_type == 'isolation_forest':
                return self._explain_anomaly_detection(input_data)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            raise
            
    def explain_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Calculate and explain feature importance."""
        try:
            if self.model_type == 'random_forest':
                return self._explain_rf_feature_importance(X, y)
            elif self.model_type == 'lstm':
                return self._explain_lstm_feature_importance(X)
            else:
                raise ValueError(f"Feature importance not supported for {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error explaining feature importance: {str(e)}")
            raise
            
    def _explain_lstm_prediction(self, input_data: np.ndarray) -> Dict:
        """Explain LSTM model prediction using attention-like mechanism."""
        try:
            # Get model predictions
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data)
                output = self.model(input_tensor)
                
            # Calculate gradients for input
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            output.backward()
            
            # Get input gradients as importance scores
            importance_scores = input_tensor.grad.numpy()
            
            # Create explanation
            explanation = {
                'prediction': output.numpy().tolist(),
                'feature_importance': {
                    name: float(score) 
                    for name, score in zip(self.feature_names, importance_scores.mean(axis=0))
                },
                'confidence': float(torch.softmax(output, dim=1).max().item()),
                'explanation_type': 'gradient_based'
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining LSTM prediction: {str(e)}")
            raise
            
    def _explain_rf_prediction(self, input_data: np.ndarray) -> Dict:
        """Explain Random Forest prediction using feature contributions."""
        try:
            # Get prediction probabilities
            probabilities = self.model.predict_proba(input_data)
            prediction = self.model.predict(input_data)
            
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create explanation
            explanation = {
                'prediction': prediction.tolist(),
                'probabilities': probabilities.tolist(),
                'feature_importance': {
                    name: float(importance)
                    for name, importance in zip(self.feature_names, importances)
                },
                'confidence': float(probabilities.max()),
                'explanation_type': 'feature_importance'
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining RF prediction: {str(e)}")
            raise
            
    def _explain_anomaly_detection(self, input_data: np.ndarray) -> Dict:
        """Explain anomaly detection results."""
        try:
            # Get anomaly scores
            scores = self.model.score_samples(input_data)
            predictions = self.model.predict(input_data)
            
            # Calculate feature-wise anomaly scores
            feature_scores = {}
            for i, feature in enumerate(self.feature_names):
                # Calculate how much each feature contributes to the anomaly score
                feature_scores[feature] = float(scores[i])
            
            # Create explanation
            explanation = {
                'anomaly_score': float(scores.mean()),
                'is_anomaly': bool(predictions[0] == -1),
                'feature_contributions': feature_scores,
                'explanation_type': 'anomaly_detection'
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining anomaly detection: {str(e)}")
            raise
            
    def _explain_rf_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Calculate detailed feature importance for Random Forest."""
        try:
            # Calculate permutation importance
            result = permutation_importance(
                self.model, X, y,
                n_repeats=10,
                random_state=42
            )
            
            # Create importance explanation
            importance = {
                'feature_importance': {
                    name: {
                        'importance': float(imp),
                        'std': float(std)
                    }
                    for name, imp, std in zip(
                        self.feature_names,
                        result.importances_mean,
                        result.importances_std
                    )
                },
                'top_features': [
                    name for name, _ in sorted(
                        zip(self.feature_names, result.importances_mean),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                ]
            }
            
            return importance
            
        except Exception as e:
            logger.error(f"Error explaining RF feature importance: {str(e)}")
            raise
            
    def _explain_lstm_feature_importance(self, X: np.ndarray) -> Dict:
        """Calculate feature importance for LSTM model."""
        try:
            # Convert input to tensor
            X_tensor = torch.FloatTensor(X)
            X_tensor.requires_grad = True
            
            # Get model output
            output = self.model(X_tensor)
            
            # Calculate gradients
            output.backward()
            
            # Get average absolute gradients as importance scores
            importance_scores = torch.abs(X_tensor.grad).mean(dim=0).numpy()
            
            # Create importance explanation
            importance = {
                'feature_importance': {
                    name: float(score)
                    for name, score in zip(self.feature_names, importance_scores)
                },
                'top_features': [
                    name for name, _ in sorted(
                        zip(self.feature_names, importance_scores),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                ]
            }
            
            return importance
            
        except Exception as e:
            logger.error(f"Error explaining LSTM feature importance: {str(e)}")
            raise 