import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataIngestion:
    def __init__(self):
        self.sensor_types = {
            'gas': ['CO', 'CH4', 'H2S'],
            'environmental': ['temperature', 'humidity'],
            'structural': ['vibration', 'pressure']
        }
        
    def generate_mock_data(self, sensor_id: str, sensor_type: str, 
                          start_time: datetime, duration_minutes: int) -> pd.DataFrame:
        """Generate mock sensor data for testing and development."""
        timestamps = pd.date_range(start=start_time, 
                                 periods=duration_minutes,
                                 freq='1min')
        
        if sensor_type in self.sensor_types['gas']:
            data = self._generate_gas_data(sensor_type, len(timestamps))
        elif sensor_type in self.sensor_types['environmental']:
            data = self._generate_environmental_data(sensor_type, len(timestamps))
        else:
            data = self._generate_structural_data(sensor_type, len(timestamps))
            
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sensor_id': sensor_id,
            'sensor_type': sensor_type,
            'value': data
        })
        
        return df
    
    def _generate_gas_data(self, gas_type: str, n_samples: int) -> np.ndarray:
        """Generate mock gas sensor data with realistic patterns."""
        if gas_type == 'CO':
            base = 5  # ppm
            noise = np.random.normal(0, 0.5, n_samples)
        elif gas_type == 'CH4':
            base = 0.5  # %
            noise = np.random.normal(0, 0.1, n_samples)
        else:  # H2S
            base = 0.1  # ppm
            noise = np.random.normal(0, 0.05, n_samples)
            
        return np.maximum(base + noise, 0)
    
    def _generate_environmental_data(self, env_type: str, n_samples: int) -> np.ndarray:
        """Generate mock environmental sensor data."""
        if env_type == 'temperature':
            base = 25  # Celsius
            noise = np.random.normal(0, 1, n_samples)
            return base + noise
        else:  # humidity
            base = 60  # %
            noise = np.random.normal(0, 5, n_samples)
            return np.clip(base + noise, 0, 100)
    
    def _generate_structural_data(self, struct_type: str, n_samples: int) -> np.ndarray:
        """Generate mock structural sensor data."""
        if struct_type == 'vibration':
            base = 0.1  # g
            noise = np.random.normal(0, 0.05, n_samples)
            return np.maximum(base + noise, 0)
        else:  # pressure
            base = 100  # kPa
            noise = np.random.normal(0, 10, n_samples)
            return np.maximum(base + noise, 0)
    
    def process_real_time_data(self, data: Dict) -> pd.DataFrame:
        """Process incoming real-time sensor data."""
        try:
            df = pd.DataFrame([data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error processing real-time data: {str(e)}")
            raise
    
    def detect_anomalies(self, df: pd.DataFrame, 
                        sensor_type: str, 
                        threshold: float = 2.0) -> pd.DataFrame:
        """Simple anomaly detection using z-score."""
        df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()
        df['is_anomaly'] = abs(df['z_score']) > threshold
        return df

# Example usage
if __name__ == "__main__":
    sensor_ingestion = SensorDataIngestion()
    
    # Generate mock data for testing
    mock_data = sensor_ingestion.generate_mock_data(
        sensor_id="SENSOR_001",
        sensor_type="CO",
        start_time=datetime.now(),
        duration_minutes=60
    )
    
    print("Generated mock data:")
    print(mock_data.head())
    
    # Example of anomaly detection
    anomalies = sensor_ingestion.detect_anomalies(mock_data, "CO")
    print("\nDetected anomalies:")
    print(anomalies[anomalies['is_anomaly']]) 