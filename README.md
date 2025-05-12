# Heptadev Challenge: Mine AI Monitoring System

A predictive monitoring system for underground mines that uses AI to forecast hazardous events and provide real-time monitoring of environmental and structural conditions.

## Project Overview

This project is a submission for the Heptadev AI Engineering Challenge, focusing on developing an AI-powered monitoring system for underground mines. The system processes real-time sensor data and uses machine learning to predict potential hazardous events, providing early warnings and detailed analysis of mine conditions.

## Key Features

- **Real-time Data Processing**: Ingests and processes data from various sensors including gas levels, temperature, humidity, vibration, and soil pressure
- **Predictive Analytics**: Uses LSTM networks for time series prediction and anomaly detection
- **3D Visualization**: Interactive 3D visualization of mine layout and sensor data
- **UDEC Integration**: Exports mine data to UDEC-compatible format for structural analysis
- **Power Management**: Optimizes sensor operations based on battery levels and network conditions
- **Interpretable AI**: Provides detailed explanations for predictions and model decisions

## System Architecture

The system is organized into the following components:

### Data Ingestion Layer
- `src/data_ingestion/sensor_data.py`: Handles real-time sensor data processing
- `src/data_ingestion/power_management.py`: Manages sensor power and sampling rates

### Machine Learning Layer
- `src/ml/predictor.py`: Implements LSTM, Isolation Forest, and Random Forest models
- `src/ml/model_explainer.py`: Provides model interpretability and explanations

### Visualization Layer
- `src/visualization/mine_visualizer.py`: 3D visualization of mine layout and sensor data
- `src/visualization/udec_integration.py`: UDEC data conversion and export
- `src/visualization/dashboard.py`: Interactive monitoring interface

### API Layer
- `src/api/main.py`: FastAPI endpoints for data processing and visualization

## Technical Stack

- **Backend**: Python, FastAPI
- **Machine Learning**: PyTorch, scikit-learn
- **Visualization**: Plotly, Dash
- **Database**: SQLAlchemy, Redis
- **Testing**: pytest

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heptadev-challenge-mine-ai.git
cd heptadev-challenge-mine-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the API server:
```bash
python src/api/main.py
```

5. Start the dashboard:
```bash
python src/visualization/dashboard.py
```

## Model Architecture

The system uses three main types of models:

1. **LSTM Network**: For time series prediction of sensor readings
2. **Isolation Forest**: For anomaly detection in sensor data
3. **Random Forest**: For hazard classification and risk assessment

## Data Sources

The system processes data from various sensors:
- Gas sensors (CO, CH4, H2S)
- Temperature and humidity sensors
- Vibration sensors
- Soil pressure sensors

## API Endpoints

- `POST /api/sensor-data`: Ingest sensor data
- `GET /api/predictions`: Get model predictions
- `GET /api/visualization`: Get 3D visualization data
- `GET /api/udec-export`: Export data to UDEC format

## Dashboard Features

- Real-time sensor data monitoring
- Predictive analytics visualization
- Battery status and power management
- Alert system for hazardous conditions

## Deployment Considerations

The system is designed to operate in low-infrastructure environments:
- Power-efficient sensor operations
- Intermittent connectivity handling
- Data compression for low bandwidth
- Configurable sampling rates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 