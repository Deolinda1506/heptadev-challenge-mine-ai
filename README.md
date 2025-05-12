Predictive Monitoring System for Underground Mines
A predictive system that leverages AI to monitor environmental and structural conditions in underground mines, providing early detection of hazards and visual insight into risk zones.

📌 Overview
The Predictive Monitoring System (PMS) is designed to enhance underground mining safety and efficiency. It uses AI-powered anomaly detection to identify hazardous patterns in real-time, such as gas build-up, temperature spikes, humidity changes, excessive vibration, and structural pressure. The system combines sensor simulation, machine learning, and 3D visualization to deliver actionable insights.

🚀 Key Features
🔧 Sensor Data Simulation
Simulates sensor readings including:

Gas levels (e.g., CO, CH₄)

Temperature & humidity

Vibration & soil pressure

Mimics the dynamics of real-world underground mining environments.

🤖 Anomaly Detection
Utilizes Isolation Forest for unsupervised detection of abnormal readings.

Flags data patterns that deviate from normal operational conditions.

📊 Risk Visualization
Provides 3D visualizations of the mine layout.

Risk zones are dynamically color-coded (e.g., red for high danger).

Displays evolving risk based on anomaly distribution.

🧠 Model Explainability
Integrates SHAP (Shapley Additive Explanations) to interpret model outputs.

Highlights which features most influenced each anomaly prediction.

Improves operator trust in AI decision-making.

🛠️ Setup and Requirements
✅ Prerequisites
Python 3.7 or higher

📦 Required Libraries
Install dependencies with:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib plotly shap joblib
📁 Project Structure
File	Description
Untitled10.ipynb	Main notebook with code for data generation, model training, and visualization.
mock_sensor_data.csv	Simulated dataset of sensor readings.
anomaly_model.pkl	Trained Isolation Forest model for detecting anomalies.
scaler.pkl	Preprocessing scaler used for feature standardization.

🧪 How to Run
1️⃣ Generate Mock Sensor Data
Simulates realistic underground mine data, saved as mock_sensor_data.csv.

2️⃣ Train Anomaly Detection Model
Preprocesses data using StandardScaler.

Trains an Isolation Forest model.

Saves model to anomaly_model.pkl and scaler to scaler.pkl.

3️⃣ Visualize Hazard Zones
Renders a 3D mine tunnel layout.

Visualizes sensor locations and real-time risk using Plotly.

4️⃣ Explain Predictions
Uses SHAP to analyze how individual features contribute to each anomaly.

Enhances transparency and decision support.

🔮 Future Enhancements
⏱️ Real-time data ingestion from physical IoT devices.

🛰️ Edge device deployment for local inference in offline environments.

📈 Time-series visualization to track how risks evolve over time.

✅ Conclusion
This project demonstrates the integration of AI and visualization for proactive mine safety management. By combining anomaly detection with intuitive 3D mapping, it provides a foundation for intelligent hazard prediction systems in challenging underground environments.

