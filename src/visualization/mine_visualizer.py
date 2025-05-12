import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MineVisualizer:
    def __init__(self):
        self.mine_layout = None
        self.sensor_positions = {}
        self.hazard_zones = {}
        
    def load_mine_layout(self, layout_file: str):
        """Load mine layout from JSON file."""
        try:
            with open(layout_file, 'r') as f:
                self.mine_layout = json.load(f)
        except Exception as e:
            logger.error(f"Error loading mine layout: {str(e)}")
            raise
    
    def add_sensor_position(self, sensor_id: str, x: float, y: float, z: float):
        """Add sensor position to the visualization."""
        self.sensor_positions[sensor_id] = {'x': x, 'y': y, 'z': z}
    
    def add_hazard_zone(self, zone_id: str, 
                       center: List[float],
                       radius: float,
                       hazard_type: str,
                       severity: float):
        """Add a hazard zone to the visualization."""
        self.hazard_zones[zone_id] = {
            'center': center,
            'radius': radius,
            'type': hazard_type,
            'severity': severity
        }
    
    def create_3d_visualization(self, 
                              sensor_data: Optional[Dict] = None,
                              show_hazards: bool = True) -> go.Figure:
        """Create 3D visualization of the mine with sensors and hazards."""
        fig = go.Figure()
        
        # Add mine tunnels
        if self.mine_layout:
            for tunnel in self.mine_layout['tunnels']:
                fig.add_trace(go.Scatter3d(
                    x=tunnel['x'],
                    y=tunnel['y'],
                    z=tunnel['z'],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    name='Tunnel'
                ))
        
        # Add sensors
        for sensor_id, pos in self.sensor_positions.items():
            color = 'blue'
            if sensor_data and sensor_id in sensor_data:
                # Color based on sensor reading
                value = sensor_data[sensor_id]['value']
                if value > sensor_data[sensor_id].get('threshold', 0):
                    color = 'red'
            
            fig.add_trace(go.Scatter3d(
                x=[pos['x']],
                y=[pos['y']],
                z=[pos['z']],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle'
                ),
                name=f'Sensor {sensor_id}'
            ))
        
        # Add hazard zones
        if show_hazards:
            for zone_id, zone in self.hazard_zones.items():
                # Create sphere for hazard zone
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                
                x = zone['center'][0] + zone['radius'] * np.outer(np.cos(u), np.sin(v))
                y = zone['center'][1] + zone['radius'] * np.outer(np.sin(u), np.sin(v))
                z = zone['center'][2] + zone['radius'] * np.outer(np.ones(np.size(u)), np.cos(v))
                
                # Color based on severity
                color = f'rgba(255, {int(255 * (1 - zone["severity"]))}, 0, 0.3)'
                
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    opacity=0.3,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=f'Hazard Zone {zone_id}'
                ))
        
        # Update layout
        fig.update_layout(
            title='Mine Monitoring System - 3D Visualization',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            showlegend=True
        )
        
        return fig
    
    def create_time_series_plot(self, 
                              sensor_data: pd.DataFrame,
                              sensor_id: str) -> go.Figure:
        """Create time series plot for a specific sensor."""
        sensor_df = sensor_data[sensor_data['sensor_id'] == sensor_id]
        
        fig = go.Figure()
        
        # Add main value line
        fig.add_trace(go.Scatter(
            x=sensor_df['timestamp'],
            y=sensor_df['value'],
            mode='lines',
            name='Sensor Value'
        ))
        
        # Add anomaly points if available
        if 'is_anomaly' in sensor_df.columns:
            anomalies = sensor_df[sensor_df['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['value'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='x'
                ),
                name='Anomalies'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Sensor {sensor_id} - Time Series',
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=True
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create sample mine layout
    sample_layout = {
        'tunnels': [
            {'x': [0, 100], 'y': [0, 0], 'z': [0, 0]},
            {'x': [100, 100], 'y': [0, 50], 'z': [0, 0]},
            {'x': [100, 0], 'y': [50, 50], 'z': [0, 0]}
        ]
    }
    
    # Save sample layout
    with open('sample_mine_layout.json', 'w') as f:
        json.dump(sample_layout, f)
    
    # Initialize visualizer
    visualizer = MineVisualizer()
    visualizer.load_mine_layout('sample_mine_layout.json')
    
    # Add sample sensors
    visualizer.add_sensor_position('SENSOR_001', 50, 0, 0)
    visualizer.add_sensor_position('SENSOR_002', 100, 25, 0)
    
    # Add sample hazard zone
    visualizer.add_hazard_zone('HAZARD_001', [75, 25, 0], 10, 'gas_leak', 0.8)
    
    # Create visualization
    fig = visualizer.create_3d_visualization()
    fig.show() 