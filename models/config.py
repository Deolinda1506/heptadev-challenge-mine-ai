"""
Configuration parameters for the mine monitoring system models.
"""

# LSTM Model Configuration
LSTM_CONFIG = {
    'input_size': 1,
    'hidden_size': 64,
    'num_layers': 2,
    'output_size': 1,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'sequence_length': 10
}

# Anomaly Detection Configuration
ANOMALY_DETECTOR_CONFIG = {
    'contamination': 0.1,
    'n_estimators': 100,
    'random_state': 42
}

# Hazard Classification Configuration
HAZARD_CLASSIFIER_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1
}

# Sensor Thresholds
SENSOR_THRESHOLDS = {
    'CO': 50,  # ppm
    'CH4': 1.0,  # %
    'H2S': 10,  # ppm
    'temperature': 35,  # Celsius
    'humidity': 90,  # %
    'vibration': 0.5,  # g
    'pressure': 150  # kPa
}

# Alert Levels
ALERT_LEVELS = {
    'LOW': 0.3,
    'MEDIUM': 0.6,
    'HIGH': 0.8,
    'CRITICAL': 0.9
}

# Model Paths
MODEL_PATHS = {
    'lstm': 'models/saved/lstm_model.pth',
    'anomaly_detector': 'models/saved/anomaly_detector.pkl',
    'hazard_classifier': 'models/saved/hazard_classifier.pkl'
}

# Training Configuration
TRAINING_CONFIG = {
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'min_delta': 0.001
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'steps_ahead': 5,
    'confidence_threshold': 0.8
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'update_interval': 5,  # seconds
    'max_data_points': 1000,
    'color_scheme': {
        'normal': 'blue',
        'warning': 'yellow',
        'critical': 'red'
    }
} 