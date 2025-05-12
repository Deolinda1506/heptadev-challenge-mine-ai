Predictive Monitoring System for Underground Mines
🚀 Overview
The Predictive Monitoring System for underground mines is designed to enhance safety and operational efficiency. By leveraging machine learning and artificial intelligence, this system continuously monitors critical environmental and structural factors in the mine, such as gas levels, temperature, humidity, vibration, and pressure. It employs anomaly detection techniques to identify potential risks and visualizes hazard zones within the mine for informed decision-making.

Key Features 🌟
1. Mock Sensor Data Generation 🔢
Simulates sensor data for monitoring factors like gas levels, temperature, humidity, vibration, and pressure.

Injects anomalies to simulate potential hazards, offering a realistic dataset for testing the anomaly detection model.

2. Anomaly Detection Model 🧠
Uses the Isolation Forest algorithm to detect outliers in the sensor data.

Identifies patterns that may indicate abnormal or risky conditions, ensuring safety through early detection.

3. 3D Hazard Visualization 🌍
Generates an interactive 3D visualization using Plotly to map out the risk zones within the mine tunnel.

Each sensor's location and risk level are displayed, helping to pinpoint hazardous areas more effectively.

4. UDEC-style Tunnel Visualization 🛠️
Provides a 2D simulation of the mine tunnel with risk levels visually represented by colors.

A hazard map shows sensor locations and helps to quickly assess risky areas.

Steps and Components 🛠️
1. Generate Mock Sensor Data
The system generates synthetic sensor data that mimics real-world conditions.

Anomalies are injected periodically to simulate dangerous situations that could be detected in a real mine environment.

2. Train Anomaly Detection Model
The Isolation Forest algorithm is applied to the sensor data to identify unusual patterns and potential risks.

The trained model and scaler are saved for future anomaly detection in real-time scenarios.

3. Simulated UDEC-style Tunnel Visualization
A 2D hazard map is created, representing sensor locations and their corresponding risk levels.

Risk zones are color-coded to provide a quick and clear assessment of hazardous areas.

4. 3D Hazard Visualization in a Mine Tunnel
The system generates a 3D visualization of the mine tunnel, where each sensor’s risk is shown using color gradients.

Depth, width, and length of the tunnel are represented, providing a comprehensive view of the mine’s risk zones.

Installation Requirements 🧰
To get started, you’ll need the following Python libraries:

pandas

numpy

matplotlib

scikit-learn

joblib

plotly

shap

You can install them all at once by running:

bash
Copy
Edit
pip install pandas numpy matplotlib scikit-learn joblib plotly shap
Next Steps 🌱
Enhance Anomaly Detection: Explore other machine learning algorithms, such as Autoencoders or clustering models, to improve anomaly detection.

Real-time Deployment: Implement the system to monitor live data from sensors, automatically flagging anomalies.

Dashboard Integration: Build a real-time dashboard (using Streamlit or Dash) to visualize ongoing sensor data and anomalies for faster, data-driven decision-making.

License 📜
This project is licensed under the MIT License. For more details, see the LICENSE file.

