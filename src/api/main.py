from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import json
import logging
from datetime import datetime
import pandas as pd

from ..data_ingestion.sensor_data import SensorDataIngestion
from ..ml.predictor import MinePredictor
from ..visualization.mine_visualizer import MineVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mine Monitoring System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
sensor_ingestion = SensorDataIngestion()
predictor = MinePredictor()
visualizer = MineVisualizer()

# Load mine layout
try:
    visualizer.load_mine_layout('sample_mine_layout.json')
except Exception as e:
    logger.warning(f"Could not load mine layout: {str(e)}")

# Data models
class SensorData(BaseModel):
    sensor_id: str
    sensor_type: str
    value: float
    timestamp: datetime

class AlertConfig(BaseModel):
    sensor_id: str
    threshold: float
    alert_type: str

# In-memory storage (replace with database in production)
sensor_data_store: Dict[str, List[Dict]] = {}
alert_configs: Dict[str, AlertConfig] = {}

@app.post("/sensor-data")
async def receive_sensor_data(data: SensorData):
    """Receive and process sensor data."""
    try:
        # Process the data
        df = sensor_ingestion.process_real_time_data(data.dict())
        
        # Store the data
        if data.sensor_id not in sensor_data_store:
            sensor_data_store[data.sensor_id] = []
        sensor_data_store[data.sensor_id].append(data.dict())
        
        # Check for anomalies
        df_with_anomalies = predictor.detect_anomalies(df)
        
        # If anomaly detected, trigger alert
        if df_with_anomalies['is_anomaly'].any():
            logger.warning(f"Anomaly detected for sensor {data.sensor_id}")
            # In production, this would trigger notifications
        
        return {"status": "success", "anomaly_detected": df_with_anomalies['is_anomaly'].any()}
    
    except Exception as e:
        logger.error(f"Error processing sensor data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sensor-data/{sensor_id}")
async def get_sensor_data(sensor_id: str, limit: int = 100):
    """Get historical sensor data."""
    if sensor_id not in sensor_data_store:
        raise HTTPException(status_code=404, detail="Sensor not found")
    
    return sensor_data_store[sensor_id][-limit:]

@app.post("/alerts/config")
async def configure_alert(config: AlertConfig):
    """Configure alert thresholds for a sensor."""
    alert_configs[config.sensor_id] = config
    return {"status": "success"}

@app.get("/predictions/{sensor_id}")
async def get_predictions(sensor_id: str, steps_ahead: int = 5):
    """Get predictions for a sensor."""
    if sensor_id not in sensor_data_store:
        raise HTTPException(status_code=404, detail="Sensor not found")
    
    # Convert stored data to DataFrame
    df = pd.DataFrame(sensor_data_store[sensor_id])
    
    try:
        # Train model if not already trained
        if not predictor.lstm_model:
            predictor.train_lstm(df)
        
        # Get predictions
        predictions = predictor.predict_next_values(df, steps_ahead)
        
        return {
            "sensor_id": sensor_id,
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualization/3d")
async def get_3d_visualization():
    """Get 3D visualization of the mine."""
    try:
        # Convert stored data to format expected by visualizer
        sensor_data = {}
        for sensor_id, data in sensor_data_store.items():
            if data:
                latest_data = data[-1]
                sensor_data[sensor_id] = {
                    'value': latest_data['value'],
                    'threshold': alert_configs.get(sensor_id, AlertConfig(
                        sensor_id=sensor_id,
                        threshold=0,
                        alert_type="default"
                    )).threshold
                }
        
        # Create visualization
        fig = visualizer.create_3d_visualization(sensor_data)
        
        return {"plot": fig.to_json()}
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualization/time-series/{sensor_id}")
async def get_time_series_visualization(sensor_id: str):
    """Get time series visualization for a sensor."""
    if sensor_id not in sensor_data_store:
        raise HTTPException(status_code=404, detail="Sensor not found")
    
    try:
        # Convert stored data to DataFrame
        df = pd.DataFrame(sensor_data_store[sensor_id])
        
        # Create visualization
        fig = visualizer.create_time_series_plot(df, sensor_id)
        
        return {"plot": fig.to_json()}
    
    except Exception as e:
        logger.error(f"Error generating time series visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 