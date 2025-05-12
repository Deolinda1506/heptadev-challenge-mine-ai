import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PowerManager:
    def __init__(self, config: Dict):
        self.config = config
        self.sensor_states = {}
        self.battery_levels = {}
        self.last_update = {}
        self.alert_threshold = config.get('battery_alert_threshold', 20)  # 20% battery level
        self.critical_threshold = config.get('battery_critical_threshold', 10)  # 10% battery level
        
    def update_sensor_state(self, sensor_id: str, battery_level: float, 
                          last_reading: datetime) -> Dict:
        """Update sensor state and calculate optimal sampling rate."""
        try:
            self.battery_levels[sensor_id] = battery_level
            self.last_update[sensor_id] = last_reading
            
            # Calculate optimal sampling rate based on battery level
            sampling_rate = self._calculate_sampling_rate(sensor_id, battery_level)
            
            # Update sensor state
            self.sensor_states[sensor_id] = {
                'battery_level': battery_level,
                'sampling_rate': sampling_rate,
                'last_update': last_reading,
                'status': self._get_sensor_status(battery_level)
            }
            
            return self.sensor_states[sensor_id]
            
        except Exception as e:
            logger.error(f"Error updating sensor state: {str(e)}")
            raise
            
    def get_sensor_config(self, sensor_id: str) -> Dict:
        """Get optimal configuration for a sensor."""
        try:
            if sensor_id not in self.sensor_states:
                raise ValueError(f"Sensor {sensor_id} not found")
                
            state = self.sensor_states[sensor_id]
            battery_level = state['battery_level']
            
            # Get base configuration
            config = self.config.get('sensor_configs', {}).get(sensor_id, {})
            
            # Adjust configuration based on battery level
            if battery_level <= self.critical_threshold:
                # Critical battery - minimal operation
                config.update({
                    'sampling_rate': self._get_minimal_sampling_rate(sensor_id),
                    'transmission_power': 'low',
                    'data_compression': 'high',
                    'sleep_mode': True
                })
            elif battery_level <= self.alert_threshold:
                # Low battery - reduced operation
                config.update({
                    'sampling_rate': state['sampling_rate'],
                    'transmission_power': 'medium',
                    'data_compression': 'medium',
                    'sleep_mode': False
                })
            else:
                # Normal operation
                config.update({
                    'sampling_rate': state['sampling_rate'],
                    'transmission_power': 'high',
                    'data_compression': 'low',
                    'sleep_mode': False
                })
                
            return config
            
        except Exception as e:
            logger.error(f"Error getting sensor config: {str(e)}")
            raise
            
    def get_battery_alerts(self) -> List[Dict]:
        """Get alerts for sensors with low battery."""
        try:
            alerts = []
            for sensor_id, state in self.sensor_states.items():
                battery_level = state['battery_level']
                
                if battery_level <= self.critical_threshold:
                    alerts.append({
                        'sensor_id': sensor_id,
                        'level': 'critical',
                        'battery_level': battery_level,
                        'message': f"Critical battery level for sensor {sensor_id}"
                    })
                elif battery_level <= self.alert_threshold:
                    alerts.append({
                        'sensor_id': sensor_id,
                        'level': 'warning',
                        'battery_level': battery_level,
                        'message': f"Low battery level for sensor {sensor_id}"
                    })
                    
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting battery alerts: {str(e)}")
            raise
            
    def _calculate_sampling_rate(self, sensor_id: str, battery_level: float) -> float:
        """Calculate optimal sampling rate based on battery level."""
        try:
            # Get base sampling rate from config
            base_rate = self.config.get('sensor_configs', {}).get(sensor_id, {}).get('base_sampling_rate', 1.0)
            
            # Adjust sampling rate based on battery level
            if battery_level <= self.critical_threshold:
                return base_rate * 0.25  # 25% of base rate
            elif battery_level <= self.alert_threshold:
                return base_rate * 0.5   # 50% of base rate
            else:
                return base_rate
                
        except Exception as e:
            logger.error(f"Error calculating sampling rate: {str(e)}")
            raise
            
    def _get_minimal_sampling_rate(self, sensor_id: str) -> float:
        """Get minimal acceptable sampling rate for a sensor."""
        try:
            # Get minimal rate from config or use default
            return self.config.get('sensor_configs', {}).get(sensor_id, {}).get('minimal_sampling_rate', 0.1)
            
        except Exception as e:
            logger.error(f"Error getting minimal sampling rate: {str(e)}")
            raise
            
    def _get_sensor_status(self, battery_level: float) -> str:
        """Get sensor status based on battery level."""
        if battery_level <= self.critical_threshold:
            return 'critical'
        elif battery_level <= self.alert_threshold:
            return 'warning'
        else:
            return 'normal'
            
    def estimate_battery_life(self, sensor_id: str) -> Dict:
        """Estimate remaining battery life for a sensor."""
        try:
            if sensor_id not in self.sensor_states:
                raise ValueError(f"Sensor {sensor_id} not found")
                
            state = self.sensor_states[sensor_id]
            battery_level = state['battery_level']
            current_rate = state['sampling_rate']
            
            # Get power consumption rates from config
            power_config = self.config.get('power_consumption', {})
            base_consumption = power_config.get('base_consumption', 1.0)
            sampling_consumption = power_config.get('sampling_consumption', 0.1)
            transmission_consumption = power_config.get('transmission_consumption', 0.2)
            
            # Calculate total power consumption per hour
            hourly_consumption = (
                base_consumption +
                (sampling_consumption * current_rate) +
                (transmission_consumption * current_rate)
            )
            
            # Estimate remaining hours
            remaining_hours = (battery_level / 100) / hourly_consumption
            
            return {
                'sensor_id': sensor_id,
                'battery_level': battery_level,
                'current_consumption': hourly_consumption,
                'estimated_hours_remaining': remaining_hours,
                'estimated_days_remaining': remaining_hours / 24
            }
            
        except Exception as e:
            logger.error(f"Error estimating battery life: {str(e)}")
            raise 