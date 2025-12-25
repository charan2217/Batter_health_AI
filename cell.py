import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BatteryCell:
    """A class representing a single battery cell in the pack."""
    _simulation_time = 0.0  # Class variable to track simulation time
    
    @classmethod
    def update_simulation_time(cls, dt: float) -> None:
        """Update the simulation time for all cells."""
        cls._simulation_time += dt
        
    def _get_simulation_time(self) -> float:
        """Get the current simulation time."""
        return self._simulation_time
        
    def __init__(self, id: int, soc: float = 0.5, temperature: float = 25.0, position: Tuple[int, int] = (0, 0)):
        self.id = id
        self.soc = np.clip(soc, 0.0, 1.0)
        self.temperature = temperature
        self.voltage = 3.7  # Nominal voltage
        self.current = 5.0  # Start with higher current to generate more heat
        self.capacity = 3.6  # Ah
        self.internal_resistance = 0.05  # Increased internal resistance for more heat generation
        self.heat_capacity = 1000  # J/kg*K (slightly reduced to make temperature changes more noticeable)
        self.mass = 0.05  # kg
        self.position = position
        self.heat_transfer_coeff = 3.0  # Reduced heat transfer to make cooling more challenging
        self.surface_area = 0.01  # m^2
        self.aging_factor = 1.0  # Increases with time to simulate battery degradation
        self.heat_generation_multiplier = 1.0  # Can be increased to simulate high-load conditions

    def update(self, dt: float, ambient_temp: float) -> None:
        """Update the cell's state for a time step with extreme thermal dynamics.
        
        Args:
            dt: Time step in seconds
            ambient_temp: Ambient temperature in °C
        """
        # More aggressive heat generation with higher exponent and multiplier
        i2r_heat = (self.current ** 2.8) * self.internal_resistance * 4.0
        
        # Calculate SOC effect (more pronounced at high and low SOC)
        # Ensure we don't get complex numbers with np.clip
        soc = np.clip(self.soc, 0.0, 1.0)
        soc_effect = 0.5 * (1.0 - np.cos(2 * np.pi * soc))  # 0-1 range
        
        # Stronger time-based variation with faster cycles
        time_factor = 1.0 + 0.7 * np.sin(self._get_simulation_time() / 15.0)
        
        # More aggressive random variation
        noise = 1.0 + np.random.normal(0, 0.15)
        
        # Calculate heat with stronger effects, ensuring no complex numbers
        heat_generated = float(abs(i2r_heat * (1.8 + soc_effect) * time_factor * noise * self.heat_generation_multiplier))
        heat_generated = max(0, heat_generated)  # Ensure non-negative
        
        # Temperature difference for cooling calculation
        temp_diff = self.temperature - ambient_temp
        
        # More gradual cooling response - less aggressive at lower temps, stronger at higher temps
        # Use abs() to ensure we don't get complex numbers with fractional powers
        temp_ratio = abs(temp_diff) / 12.0
        cooling_effectiveness = 0.02 * (1.0 + (temp_ratio ** 2.5))
        cooling = cooling_effectiveness * abs(temp_diff) * (1 if temp_diff >= 0 else -1)
        
        # Adjust thermal mass for faster response
        thermal_mass = self.heat_capacity * self.mass * 0.7
        
        # Calculate temperature change - heat rises fast, cools slower
        if heat_generated > cooling:
            # When heating, use full delta
            delta_temp = (heat_generated - cooling) * dt / thermal_mass
        else:
            # When cooling, reduce the cooling rate by 40%
            delta_temp = (heat_generated - cooling * 0.6) * dt / thermal_mass
        
        self.temperature += delta_temp
        self.temperature = max(ambient_temp - 0.5, min(self.temperature, 100.0))  # Cap at 100°C
        
        # Update SOC based on current (simplified)
        if self.current != 0:
            delta_soc = (self.current * dt) / (self.capacity * 3600)  # 3600 seconds in an hour
            self.soc = np.clip(self.soc - delta_soc, 0.0, 1.0)
        
        # Update voltage based on SOC and temperature (simplified model)
        soc_voltage = 2.5 + 2.0 * float(self.soc)  # Linear approximation, ensure float
        temp_effect = 0.01 * float(self.temperature - 25)  # 10mV/°C temp coefficient, ensure float
        self.voltage = float(soc_voltage + temp_effect)  # Ensure float for JSON serialization
        
    def get_state(self) -> dict:
        """Return the current state of the cell."""
        return {
            'id': self.id,
            'soc': self.soc,
            'voltage': self.voltage,
            'current': self.current,
            'temperature': self.temperature,
            'position': self.position
        }
