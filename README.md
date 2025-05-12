
Predictive Monitoring System for Underground Mines
Overview
The Predictive Monitoring System (PMS) for Underground Mines is designed to improve worker safety and operational efficiency by leveraging AI and machine learning to detect hazardous conditions in real-time. The system monitors factors like gas build-up, temperature fluctuations, humidity changes, vibration, and structural integrity, while providing actionable insights through anomaly detection and hazard visualization.

Key Features
Sensor Data Generation:

Simulated sensor data from various sources (gas levels, temperature, humidity, vibration, pressure) to mimic the behavior of sensors in an underground mine.

Anomaly Detection:

Uses machine learning models, such as Isolation Forest, to identify anomalous sensor readings that could indicate potential hazards.

Risk Visualization:

3D visualizations of mine tunnels with dynamic risk maps, representing different risk levels, to help operators visualize the hazard zones and predict risk evolutions.

Setup and Requirements
Prerequisites
Python 3.7 or higher.

Required libraries: pandas, numpy, scikit-learn, matplotlib, plotly, shap, and joblib.

You can install the required dependencies using the following command:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib plotly shap joblib
Project Structure
Untitled10.ipynb: This Jupyter notebook contains the entire pipeline for generating data, training the model, and visualizing the results.

mock_sensor_data.csv: Simulated sensor data file containing gas levels, temperature, humidity, vibration, and pressure readings.

anomaly_model.pkl: The trained machine learning model for anomaly detection.

scaler.pkl: The preprocessing scaler used to standardize the sensor data before model training.

Running the Project
1. Generate Mock Sensor Data
This step generates synthetic sensor data, including anomalies, to simulate real-world mining conditions. The data is saved in mock_sensor_data.csv, which is later used to train the anomaly detection model.

2. Train the Anomaly Detection Model
An Isolation Forest model is trained on the generated data to detect anomalies that may indicate hazardous conditions (e.g., gas leaks or excessive vibrations). The trained model is saved as anomaly_model.pkl, and the data scaling method used in preprocessing is saved as scaler.pkl.

3. Visualize Hazard Zones
The system provides 3D visualizations of risk zones in the underground mine using Plotly. The simulation displays the locations of sensors and the associated risk levels, with color coding to indicate the severity of the risks (e.g., red indicates higher danger).

4. Model Explainability
The SHAP (Shapley Additive Explanations) library is used to explain the model’s predictions, highlighting which sensor readings most influence the detection of anomalies. This improves the transparency of the AI system and increases operator trust in the model’s decision-making.

Future Enhancements
Integration of real-time sensor data for continuous hazard monitoring in mines.

Deployment of the anomaly detection model on edge devices for real-time processing without relying on cloud infrastructure.

Improved 3D visualizations to incorporate time-series data and track the evolution of risk over time.

Conclusion
This project showcases the application of AI and machine learning to underground mining safety. By detecting hazardous conditions early and visualizing risk zones in 3D, the system helps enhance worker safety, minimize downtime, and improve operational efficiency in mining environments.
