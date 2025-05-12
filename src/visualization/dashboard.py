import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import requests
import json

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# API endpoint
API_URL = "http://localhost:8000"

# Layout
app.layout = html.Div([
    html.H1("Mine Monitoring System Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # 3D Visualization
    html.Div([
        html.H2("3D Mine Visualization", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        dcc.Graph(id='3d-visualization'),
        dcc.Interval(
            id='3d-interval',
            interval=5*1000,  # Update every 5 seconds
            n_intervals=0
        )
    ]),
    
    # Sensor Selection and Time Series
    html.Div([
        html.H2("Sensor Data", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        dcc.Dropdown(
            id='sensor-dropdown',
            options=[],  # Will be populated dynamically
            value=None,
            placeholder="Select a sensor"
        ),
        dcc.Graph(id='time-series-plot'),
        dcc.Interval(
            id='time-series-interval',
            interval=1*1000,  # Update every second
            n_intervals=0
        )
    ]),
    
    # Predictions
    html.Div([
        html.H2("Predictions", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        dcc.Graph(id='predictions-plot'),
        dcc.Interval(
            id='predictions-interval',
            interval=10*1000,  # Update every 10 seconds
            n_intervals=0
        )
    ]),
    
    # Alert Configuration
    html.Div([
        html.H2("Alert Configuration", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Div([
            html.Label("Threshold:"),
            dcc.Input(
                id='threshold-input',
                type='number',
                value=0,
                style={'margin': '10px'}
            ),
            html.Button('Update Alert', id='update-alert-button',
                       style={'margin': '10px'})
        ], style={'textAlign': 'center'})
    ])
])

# Callbacks
@app.callback(
    Output('3d-visualization', 'figure'),
    Input('3d-interval', 'n_intervals')
)
def update_3d_visualization(n):
    try:
        response = requests.get(f"{API_URL}/visualization/3d")
        if response.status_code == 200:
            return json.loads(response.json()['plot'])
    except Exception as e:
        print(f"Error updating 3D visualization: {str(e)}")
    
    # Return empty figure if there's an error
    return go.Figure()

@app.callback(
    Output('sensor-dropdown', 'options'),
    Input('3d-interval', 'n_intervals')
)
def update_sensor_dropdown(n):
    try:
        response = requests.get(f"{API_URL}/sensor-data")
        if response.status_code == 200:
            sensors = response.json()
            return [{'label': sensor_id, 'value': sensor_id} 
                   for sensor_id in sensors.keys()]
    except Exception as e:
        print(f"Error updating sensor dropdown: {str(e)}")
    
    return []

@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('sensor-dropdown', 'value'),
     Input('time-series-interval', 'n_intervals')]
)
def update_time_series(sensor_id, n):
    if not sensor_id:
        return go.Figure()
    
    try:
        response = requests.get(f"{API_URL}/visualization/time-series/{sensor_id}")
        if response.status_code == 200:
            return json.loads(response.json()['plot'])
    except Exception as e:
        print(f"Error updating time series: {str(e)}")
    
    return go.Figure()

@app.callback(
    Output('predictions-plot', 'figure'),
    [Input('sensor-dropdown', 'value'),
     Input('predictions-interval', 'n_intervals')]
)
def update_predictions(sensor_id, n):
    if not sensor_id:
        return go.Figure()
    
    try:
        response = requests.get(f"{API_URL}/predictions/{sensor_id}")
        if response.status_code == 200:
            data = response.json()
            
            # Create predictions plot
            fig = go.Figure()
            
            # Add historical data
            historical_response = requests.get(f"{API_URL}/sensor-data/{sensor_id}")
            if historical_response.status_code == 200:
                historical_data = historical_response.json()
                timestamps = [datetime.fromisoformat(d['timestamp']) 
                            for d in historical_data]
                values = [d['value'] for d in historical_data]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name='Historical Data'
                ))
            
            # Add predictions
            future_timestamps = [
                timestamps[-1] + timedelta(minutes=i+1)
                for i in range(len(data['predictions']))
            ]
            
            fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=data['predictions'],
                mode='lines',
                name='Predictions',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title='Sensor Predictions',
                xaxis_title='Time',
                yaxis_title='Value',
                showlegend=True
            )
            
            return fig
    except Exception as e:
        print(f"Error updating predictions: {str(e)}")
    
    return go.Figure()

@app.callback(
    Output('update-alert-button', 'children'),
    [Input('update-alert-button', 'n_clicks'),
     Input('sensor-dropdown', 'value'),
     Input('threshold-input', 'value')]
)
def update_alert_config(n_clicks, sensor_id, threshold):
    if not n_clicks or not sensor_id:
        return 'Update Alert'
    
    try:
        response = requests.post(
            f"{API_URL}/alerts/config",
            json={
                'sensor_id': sensor_id,
                'threshold': threshold,
                'alert_type': 'threshold'
            }
        )
        if response.status_code == 200:
            return 'Alert Updated!'
    except Exception as e:
        print(f"Error updating alert config: {str(e)}")
    
    return 'Update Failed'

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 