import numpy as np
import json
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UDECIntegration:
    def __init__(self):
        self.geometry_data = None
        self.stress_data = None
        self.fluid_data = None
        
    def convert_mine_geometry(self, mine_layout: Dict) -> Dict:
        """Convert mine layout to UDEC-compatible geometry format."""
        try:
            udec_geometry = {
                'blocks': [],
                'joints': [],
                'boundaries': []
            }
            
            # Convert tunnels to UDEC blocks
            for tunnel in mine_layout['tunnels']:
                block = {
                    'id': tunnel.get('id', ''),
                    'vertices': self._convert_vertices(tunnel['x'], tunnel['y'], tunnel['z']),
                    'material': tunnel.get('material', 'rock'),
                    'properties': tunnel.get('properties', {})
                }
                udec_geometry['blocks'].append(block)
            
            # Add structural elements
            if 'structural_elements' in mine_layout:
                for element in mine_layout['structural_elements']:
                    joint = {
                        'id': element.get('id', ''),
                        'points': self._convert_points(element['points']),
                        'type': element.get('type', 'joint'),
                        'properties': element.get('properties', {})
                    }
                    udec_geometry['joints'].append(joint)
            
            return udec_geometry
            
        except Exception as e:
            logger.error(f"Error converting mine geometry: {str(e)}")
            raise
    
    def convert_stress_data(self, sensor_data: Dict) -> Dict:
        """Convert sensor data to UDEC stress analysis format."""
        try:
            udec_stress = {
                'stress_tensors': [],
                'displacements': [],
                'strains': []
            }
            
            # Convert pressure and vibration data to stress tensors
            for sensor_id, data in sensor_data.items():
                if data['sensor_type'] in ['pressure', 'vibration']:
                    stress_tensor = self._calculate_stress_tensor(data)
                    udec_stress['stress_tensors'].append({
                        'location': data['location'],
                        'tensor': stress_tensor,
                        'timestamp': data['timestamp']
                    })
            
            return udec_stress
            
        except Exception as e:
            logger.error(f"Error converting stress data: {str(e)}")
            raise
    
    def convert_fluid_data(self, sensor_data: Dict) -> Dict:
        """Convert sensor data to UDEC fluid flow format."""
        try:
            udec_fluid = {
                'pressure_points': [],
                'flow_rates': [],
                'permeability': []
            }
            
            # Convert gas and humidity data to fluid flow parameters
            for sensor_id, data in sensor_data.items():
                if data['sensor_type'] in ['gas', 'humidity']:
                    fluid_data = self._calculate_fluid_parameters(data)
                    udec_fluid['pressure_points'].append({
                        'location': data['location'],
                        'pressure': fluid_data['pressure'],
                        'timestamp': data['timestamp']
                    })
            
            return udec_fluid
            
        except Exception as e:
            logger.error(f"Error converting fluid data: {str(e)}")
            raise
    
    def export_to_udec(self, mine_data: Dict) -> str:
        """Export complete mine data to UDEC format."""
        try:
            udec_data = {
                'geometry': self.convert_mine_geometry(mine_data['layout']),
                'stress_analysis': self.convert_stress_data(mine_data['sensor_data']),
                'fluid_flow': self.convert_fluid_data(mine_data['sensor_data']),
                'metadata': {
                    'timestamp': mine_data.get('timestamp', ''),
                    'version': '1.0',
                    'format': 'udec'
                }
            }
            
            return json.dumps(udec_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting to UDEC: {str(e)}")
            raise
    
    def _convert_vertices(self, x: List[float], y: List[float], z: List[float]) -> List[Dict]:
        """Convert coordinate lists to vertex format."""
        return [{'x': xi, 'y': yi, 'z': zi} for xi, yi, zi in zip(x, y, z)]
    
    def _convert_points(self, points: List[Dict]) -> List[Dict]:
        """Convert point data to UDEC format."""
        return [{'x': p['x'], 'y': p['y'], 'z': p['z']} for p in points]
    
    def _calculate_stress_tensor(self, data: Dict) -> np.ndarray:
        """Calculate stress tensor from sensor data."""
        # Simplified stress tensor calculation
        pressure = data.get('value', 0)
        return np.array([
            [pressure, 0, 0],
            [0, pressure, 0],
            [0, 0, pressure]
        ])
    
    def _calculate_fluid_parameters(self, data: Dict) -> Dict:
        """Calculate fluid flow parameters from sensor data."""
        # Simplified fluid parameter calculation
        return {
            'pressure': data.get('value', 0),
            'flow_rate': data.get('value', 0) * 0.1,  # Simplified conversion
            'permeability': 1.0  # Default value
        } 