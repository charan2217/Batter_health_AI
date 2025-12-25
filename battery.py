# models/battery.py
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import tensorflow as tf
from dataclasses import dataclass, field
from .cell import BatteryCell
from .pinn import PINN as PINNModel
from .risk_lstm import RiskLSTM
from .rl_agent import RLAgent, ActionLog
import time
import json

@dataclass
class BatteryPack:
    """A class representing a battery pack with multiple cells."""
    rows: int = 10
    cols: int = 10
    ambient_temp: float = 25.0
    use_pinn: bool = True
    pinn_model_path: Optional[str] = None

    def __post_init__(self):
        self.cells: List[List[BatteryCell]] = []
        self.time_elapsed: float = 0.0
        self.pinn_models: Dict[int, PINNModel] = {}
        self._initialize_cells()
        if self.use_pinn:
            self._initialize_pinn_models()
        # --- LSTM for risk ---
        self.risk_lstm = RiskLSTM(input_dim=4, seq_len=60)
        self.risk_history = []  # Store last 60 steps of [avg_temp, avg_current, avg_soc, ambient_temp]
        self.risk_score = 0.0
        # --- RL Agent for cooling ---
        self.rl_agent = RLAgent(temp_threshold=45.0)  # Slightly lower threshold for proactive cooling
        # --- Prediction and status tracking ---
        self.last_prediction = None
        self.prediction_history = []
        self.cooling_active = False
        self.last_cooling_state_change = 0
        self.prediction_messages = []
        self.cooling_animation_state = 0  # 0: off, 1: starting, 2: on, 3: stopping
        # --- Temperature tracking ---
        self.temperature_history = []  # Track temperature over time for trend analysis
        self.max_temp_reached = self.ambient_temp  # Track maximum temperature reached

    def _initialize_cells(self):
        """Initialize the battery cells in a grid."""
        cell_id = 0
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                soc = np.random.normal(0.5, 0.05)
                temp = np.random.normal(25.0, 0.5)
                cell = BatteryCell(
                    id=cell_id,
                    soc=np.clip(soc, 0.1, 0.9),
                    temperature=temp,
                    position=(i, j)
                )
                row.append(cell)
                cell_id += 1
            self.cells.append(row)

    def _initialize_pinn_models(self):
        """Initialize PINN models for temperature prediction."""
        if not self.use_pinn:
            return
            
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.cells[i][j]
                if self.pinn_model_path:
                    try:
                        self.pinn_models[cell.id] = PINNModel.load(self.pinn_model_path)
                    except:
                        print(f"Failed to load PINN model from {self.pinn_model_path}")
                        self.pinn_models[cell.id] = PINNModel()
                else:
                    self.pinn_models[cell.id] = PINNModel()

    def update(self, dt: float):
        """Update all cells in the battery pack with optimized thermal dynamics."""
        # Update simulation time for all cells
        BatteryCell.update_simulation_time(dt)
        
        # Enhanced dynamic load variations with multiple time scales
        # Add a slow oscillation between high and low load periods (2-3 minutes cycle)
        slow_oscillation = 0.5 * (1 + np.sin(2 * np.pi * self.time_elapsed / 180.0))  # Slower cycle (3 minutes)
        
        # Long cycle (3-4 minutes) for major temperature swings
        slow_cycle = 2.5 * np.sin(2 * np.pi * self.time_elapsed / 210.0)  # ~3.5 minute cycle
        
        # Medium cycle (45-60 seconds) for moderate variations
        med_cycle = 1.8 * np.sin(2 * np.pi * self.time_elapsed / 50.0)  # ~50 second cycle
        
        # Short cycle (5-10 seconds) for rapid fluctuations
        fast_cycle = 1.2 * np.sin(2 * np.pi * self.time_elapsed / 8.0)  # 8 second cycle
        
        # Add some randomness that changes over time
        noise = 0.6 * np.sin(self.time_elapsed * 0.3) * np.random.normal(0, 0.3)
        
        # Combine all components with different weights
        combined = (
            1.5 * slow_cycle +  # Strong emphasis on slow cycle
            1.0 * med_cycle +   # Medium emphasis on medium cycle
            0.5 * fast_cycle +  # Less emphasis on fast cycle
            noise
        )
        
        # Use sigmoid to create more pronounced high/low periods
        sigmoid = 1 / (1 + np.exp(-0.5 * combined))
        
        # Map to load factor with wider dynamic range (1.0 to 8.0)
        # This creates more pronounced periods of high and low load
        load_factor = 1.0 + 7.0 * sigmoid  # Range from 1.0 to 8.0
        
        # Add a cooling period where load drops significantly
        cooling_period = 0.5 * (1 + np.sin(2 * np.pi * self.time_elapsed / 300.0))  # 5 minute cycle
        if cooling_period < 0.3:  # About 30% of the time in cooling mode
            load_factor = max(1.0, load_factor * 0.4)  # Reduce load during cooling periods
        
        # Current variations based on load with dynamic scaling
        time_variation = 10.0 * load_factor
        
        # Calculate center positions for hot spot
        center_row, center_col = self.rows // 2, self.cols // 2
        
        # Pre-calculate some values
        rows_cols_sum = self.rows + self.cols
        
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                # Calculate distance from center for hot spot effect
                dist_from_center = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                
                # Enhanced position factor with stronger hot spot effect
                hot_spot_strength = 1.8  # Increased from 1.2
                hot_spot_decay = 2.5     # More focused hot spot
                position_factor = hot_spot_strength + 0.8 * np.exp(-dist_from_center / hot_spot_decay)
                
                # Add more dynamic current noise
                current_noise = np.random.normal(0, 0.6)  # Increased noise
                
                # Set current with enhanced variations
                cell.current = 12.0 + time_variation * position_factor + current_noise
                
                # Position-based heat variation (hotter in center)
                distance = np.sqrt((i - self.rows//2)**2 + (j - self.cols//2)**2)
                hot_spot_factor = max(0.2, 1.0 - (distance / (self.rows//2)))
                
                # Time-based effects with different periods
                # Very slow variation (2-3 minutes)
                slow_effect = 0.8 * (1 + np.sin(2 * np.pi * self.time_elapsed / 150.0))
                
                # Medium variation (30-40 seconds)
                med_effect = 0.6 * (1 + np.sin(2 * np.pi * self.time_elapsed / 35.0))
                
                # Fast variation (5-10 seconds)
                fast_effect = 0.4 * (1 + np.sin(2 * np.pi * self.time_elapsed / 7.5))
                
                # Random fluctuations that change gradually
                random_effect = 0.9 + 0.2 * np.sin(self.time_elapsed * 0.5) * np.random.normal(0, 0.3)
                
                # Calculate heat generation multiplier
                # Varies more with load_factor and has stronger position effect
                cell.heat_generation_multiplier = (
                    3.0 * 
                    (0.3 + 0.7 * hot_spot_factor) *  # Strong position effect
                    (0.5 + 0.5 * load_factor/8.0) *  # Vary with load
                    (1.0 + slow_effect + 0.7*med_effect + 0.3*fast_effect) *  # Time variations
                    random_effect  # Random component
                )
                
                # Update cell with optimized parameters
                cell.update(dt, self.ambient_temp)
        
        # Apply heat transfer between cells with enhanced conduction
        self._simulate_heat_transfer(dt)
        
        # Get current state and predict future temperature trend
        current_state = self.get_state()
        temp_trend, prediction_msg = self._predict_temperature_trend()
        
        # Enhance state with trend and prediction information
        current_state['temperature_trend'] = temp_trend
        current_state['time_elapsed'] = self.time_elapsed
        current_state['prediction_messages'] = self.prediction_messages
        
        # Just apply natural thermal effects
        self._apply_thermal_effects(dt)
        
        # Get current max temperature for predictions
        current_temp = max(cell.temperature for row in self.cells for cell in row)
        
        # Update temperature history for predictions (more frequent updates)
        self.temperature_history.append((self.time_elapsed, current_temp))
        if len(self.temperature_history) > 1000:
            self.temperature_history = self.temperature_history[-1000:]
        
        # Generate prediction messages before getting state
        trend, _ = self._predict_temperature_trend()
        
        # Get current state with updated predictions
        state = self.get_state()
        
        # Add thermal analysis to state
        state['thermal_analysis'] = {
            'trend': trend,
            'max_temperature': current_temp,
            'time_elapsed': self.time_elapsed
        }
        
        # Update risk history for trend analysis
        self.risk_history.append((current_temp, state['risk_level'], state['health_status']))
        if len(self.risk_history) > 100:
            self.risk_history = self.risk_history[-100:]
        
        # Ensure prediction_messages is always in state
        if not hasattr(self, 'prediction_messages'):
            self.prediction_messages = []
        state['prediction_messages'] = self.prediction_messages
        
        # Print debug info
        if self.time_elapsed % 5 < dt:  # Print every ~5 seconds
            print(f"[DEBUG] Time: {self.time_elapsed:.1f}s, Max Temp: {current_temp:.1f}°C, "
                  f"Risk: {state['risk_level']}, Health: {state['health_status']}")
            if self.prediction_messages:
                print(f"[DEBUG] Latest prediction: {self.prediction_messages[-1]['message']}")
        
        return state

    def _simulate_heat_transfer(self, dt: float):
        """Simulate heat transfer between neighboring cells with enhanced thermal modeling."""
        # First, calculate all temperature changes without applying them
        temp_changes = np.zeros((self.rows, self.cols))
        
        # Define neighbor directions (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Calculate heat transfer for each cell
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.cells[i][j]
                
                # Check all four neighbors
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    # Check if neighbor is within bounds
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        neighbor = self.cells[ni][nj]
                        
                        # Calculate temperature difference
                        temp_diff = cell.temperature - neighbor.temperature
                        
                        # Calculate heat flow (Fourier's law)
                        heat_flow = (cell.heat_transfer_coeff + neighbor.heat_transfer_coeff) / 2 * \
                                  cell.surface_area * temp_diff * dt
                        
                        # Distribute heat flow based on cell properties
                        if heat_flow > 0:  # Heat flows from current cell to neighbor
                            heat_transfer = min(heat_flow, cell.temperature - neighbor.temperature) / 2
                            temp_changes[i][j] -= heat_transfer / (cell.heat_capacity * cell.mass)
                            temp_changes[ni][nj] += heat_transfer / (neighbor.heat_capacity * neighbor.mass)
        
        # Apply all temperature changes
        for i in range(self.rows):
            for j in range(self.cols):
                self.cells[i][j].temperature += temp_changes[i][j]

    def get_temperature_grid(self) -> np.ndarray:
        """Get the temperature of all cells as a 2D numpy array."""
        return np.array([[cell.temperature for cell in row] for row in self.cells])

    def get_voltage_grid(self) -> np.ndarray:
        """Get the voltage of all cells as a 2D numpy array."""
        return np.array([[cell.voltage for cell in row] for row in self.cells])

    def get_soc_grid(self) -> np.ndarray:
        """Get the state of charge of all cells as a 2D numpy array."""
        return np.array([[cell.soc for cell in row] for row in self.cells])

    def get_pack_voltage(self) -> float:
        """Get the total voltage of the battery pack (series connection)."""
        return np.sum(self.get_voltage_grid())

    def get_average_temperature(self) -> float:
        """Get the average temperature of all cells."""
        return np.mean(self.get_temperature_grid())

    def _predict_temperature_trend(self) -> Tuple[float, str]:
        """Predict the temperature trend and generate AI-based prediction messages with enhanced sensitivity."""
        # Initialize prediction messages list if it doesn't exist
        if not hasattr(self, 'prediction_messages'):
            self.prediction_messages = []
        
        # More frequent updates for critical conditions
        current_time = time.time()
        time_since_last = current_time - getattr(self, 'last_prediction_time', 0)
        update_interval = 1.0  # Update at least once per second
        
        # Get current temperature data
        current_temp = max(cell.temperature for row in self.cells for cell in row)
        
        # Calculate trend using recent temperature history
        if len(self.temperature_history) >= 5:
            # Use last 5 temperature readings (last 0.5 seconds at 10Hz update rate)
            temps = [t[1] for t in self.temperature_history[-5:]]
            times = [t[0] for t in self.temperature_history[-5:]]
            if len(set(times)) > 1:  # Need at least 2 different time points
                slope, _ = np.polyfit(times, temps, 1)
                trend = slope * 10  # Convert to °C/s
            else:
                trend = 0.0
        else:
            trend = 0.0
        
        # Predict temperature 15 seconds ahead
        predicted_temp = current_temp + (trend * 15)
        
        # Generate prediction message based on current state and trend
        prediction_text = ""
        severity = 'info'
        
        # Thermal runaway detection (very high temp and increasing)
        if current_temp > 50 or (predicted_temp > 55 and trend > 0.25):
            prediction_text = (f"🚨 THERMAL RUNAWAY DETECTED! 🚨\n"
                             f"Current: {current_temp:.1f}°C | "
                             f"Predicted: {predicted_temp:.1f}°C in 15s | "
                             f"Rate: +{trend:.2f}°C/s")
            severity = 'critical'
            update_interval = 0.5  # Update twice per second for critical conditions
        # Critical temperature threshold
        elif current_temp > 45 or (predicted_temp > 50 and trend > 0.2):
            prediction_text = (f"🔥 CRITICAL: Temp {current_temp:.1f}°C (Rising: +{trend:.2f}°C/s)\n"
                             f"Predicted: {predicted_temp:.1f}°C in 15s")
            severity = 'high'
        # High temperature with strong upward trend
        elif trend > 0.15 or (predicted_temp > 45 and trend > 0.1):
            prediction_text = (f"⚠️ WARNING: Temp rising to {predicted_temp:.1f}°C in 15s\n"
                             f"Current: {current_temp:.1f}°C | Rate: +{trend:.2f}°C/s")
            severity = 'high'
        # Moderate temperature with upward trend
        elif trend > 0.08 or (predicted_temp > 40 and trend > 0.05):
            prediction_text = (f"📈 Temp increasing: {current_temp:.1f}°C\n"
                             f"Trend: +{trend:.2f}°C/s | Predicted: {predicted_temp:.1f}°C")
            severity = 'medium'
        # High temperature but stable or decreasing
        elif current_temp > 40:
            prediction_text = (f"⚠️ High Temp: {current_temp:.1f}°C\n"
                             f"Trend: {'+' if trend > 0 else ''}{trend:.2f}°C/s")
            severity = 'medium'
        # Temperature decreasing significantly
        elif trend < -0.05:
            prediction_text = (f"📉 Temp decreasing: {current_temp:.1f}°C\n"
                             f"Rate: {trend:.2f}°C/s")
            severity = 'info'
        # Normal operation
        else:
            prediction_text = (f"✅ Normal: {current_temp:.1f}°C\n"
                             f"Trend: {'+' if trend > 0 else ''}{trend:.2f}°C/s")
            severity = 'info'
            update_interval = 2.0  # Less frequent updates for normal conditions
        
        # Store the prediction with timestamp
        if (not hasattr(self, 'last_prediction_time') or 
            time_since_last >= update_interval or
            not hasattr(self, 'last_prediction_text') or 
            self.last_prediction_text != prediction_text):
            
            prediction_entry = {
                'time': self.time_elapsed,
                'timestamp': current_time,
                'message': prediction_text,
                'severity': severity,
                'temperature': current_temp,
                'predicted_temp': predicted_temp,
                'trend': trend
            }
            
            # Add to prediction messages
            if not hasattr(self, 'prediction_messages'):
                self.prediction_messages = []
                
            self.prediction_messages.append(prediction_entry)
            self.last_prediction_text = prediction_text
            self.last_prediction_time = current_time
            
            # Keep only the 5 most recent predictions
            if len(self.prediction_messages) > 5:
                self.prediction_messages = self.prediction_messages[-5:]
                
            # Log to console for debugging
            time_str = f"[{self.time_elapsed:6.1f}s]"
            if severity == 'critical':
                print(f"\033[91m{time_str} {prediction_text}\033[0m")  # Red
            elif severity == 'high':
                print(f"\033[93m{time_str} {prediction_text}\033[0m")  # Yellow
            else:
                print(f"{time_str} {prediction_text}")
        
        return trend, prediction_text

    def _apply_thermal_effects(self, dt: float):
        """Apply thermal effects to all cells with dynamic cooling rates."""
        # Calculate a global cooling factor that varies over time with multiple frequencies
        slow_cooling = 0.7 + 0.3 * np.sin(2 * np.pi * self.time_elapsed / 180.0)  # 3 min cycle
        med_cooling = 0.8 + 0.2 * np.sin(2 * np.pi * self.time_elapsed / 45.0)    # 45 sec cycle
        fast_cooling = 0.9 + 0.1 * np.sin(2 * np.pi * self.time_elapsed / 10.0)    # 10 sec cycle
        cooling_variation = slow_cooling * med_cooling * fast_cooling
        
        # Add a strong cooling period that happens occasionally
        cooling_period = 0.5 * (1 + np.sin(2 * np.pi * self.time_elapsed / 300.0))  # 5 min cycle
        if cooling_period < 0.3:  # About 30% of the time in enhanced cooling
            cooling_variation *= 1.8  # Significantly increase cooling
        
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(self.cells[i]):
                # Position-based cooling (edges cool faster than center)
                distance_to_edge = min(i, self.rows-1-i, j, self.cols-1-j)
                position_factor = 0.2 + 0.8 * (distance_to_edge / (self.rows//2))  # More pronounced edge cooling
                
                # Base cooling rate varies with temperature difference (non-linear)
                temp_diff = cell.temperature - self.ambient_temp
                base_cooling = 0.05 * temp_diff * dt * (1 + 0.2 * np.log1p(temp_diff))
                
                # Apply position and time-based variations
                cooling_rate = base_cooling * position_factor * cooling_variation
                
                # Random component to cooling (small)
                cooling_rate *= (0.9 + 0.2 * np.random.random())
                
                # Apply cooling with thermal inertia (smoother changes)
                if hasattr(cell, 'last_temp'):
                    # Add some thermal inertia (resistance to change)
                    temp_change = cell.temperature - cell.last_temp
                    cooling_rate *= (1.0 - 0.3 * np.tanh(temp_change * 10))
                cell.last_temp = cell.temperature
                
                # Apply cooling
                cell.temperature = max(self.ambient_temp, cell.temperature - cooling_rate)
                
                # Ensure temperature stays within reasonable bounds
                cell.temperature = min(max(cell.temperature, self.ambient_temp), 100.0)
                
                # Update current based on temperature and position
                center_row, center_col = self.rows//2, self.cols//2
                distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                edge_factor = 1.0 + 0.4 * (distance / max(center_row, center_col))
                
                # Current varies with temperature and position
                temp_effect = 0.5 + 0.5 * np.tanh((cell.temperature - 30) / 20)  # Smooth transition
                cell.current = 0.1 + (0.3 + 0.7 * temp_effect) * edge_factor

    def get_state(self):
        """Get the current state of the battery pack with enhanced risk assessment."""
        # Get temperature data
        temp_grid = self.get_temperature_grid()
        max_temp = np.max(temp_grid)
        min_temp = np.min(temp_grid)
        avg_temp = np.mean(temp_grid)
        
        # Get voltage and current metrics
        voltages = [cell.voltage for row in self.cells for cell in row]
        voltage_imbalance = max(voltages) - min(voltages) if len(voltages) > 1 else 0.0
        currents = [abs(cell.current) for row in self.cells for cell in row]
        
        # Get temperature trend for risk assessment
        temp_trend = 0.0
        if len(self.risk_history) >= 5:
            recent_temps = [h[0] for h in self.risk_history[-5:]]
            temp_trend = np.polyfit(range(len(recent_temps)), recent_temps, 1)[0] * 10
        
        # Much more sensitive risk level calculation
        risk_level = "LOW"
        if max_temp > 42 or (max_temp > 38 and temp_trend > 0.15) or (hasattr(self, 'max_temp_reached') and self.max_temp_reached > 45):
            risk_level = "CRITICAL"
        elif max_temp > 38 or (max_temp > 35 and temp_trend > 0.1):
            risk_level = "HIGH"
        elif max_temp > 34 or (max_temp > 32 and temp_trend > 0.05):
            risk_level = "MEDIUM"
        
        # Health status with memory of past high temperatures - more sensitive to temperature increases
        if not hasattr(self, 'max_temp_reached') or max_temp > getattr(self, 'max_temp_reached', 0):
            self.max_temp_reached = max_temp
            
        # More aggressive health degradation with temperature
        if self.max_temp_reached > 45 or max_temp > 43:
            health_status = "POOR"
        elif self.max_temp_reached > 40 or max_temp > 38:
            health_status = "FAIR"
        else:
            health_status = "GOOD"
        
        # Collect cell states
        cell_states = []
        for row in self.cells:
            for cell in row:
                cell_states.append({
                    'id': cell.id,
                    'temperature': float(cell.temperature),
                    'voltage': float(cell.voltage),
                    'current': float(cell.current),
                    'soc': float(cell.soc),
                    'position': cell.position
                })
        
        # Get latest RL agent action for fan and pump levels
        fan_speed = 0.0
        pump_level = 0.0
        if hasattr(self, 'rl_agent') and hasattr(self.rl_agent, 'action_history') and self.rl_agent.action_history:
            latest_action = self.rl_agent.action_history[-1].action
            if len(latest_action) >= 2:  # Ensure we have at least fan and pump values
                fan_speed, pump_level = latest_action[0], latest_action[1]
        
        # Prepare the state dictionary
        state = {
            'cells': cell_states,
            'max_temperature': float(max_temp),
            'min_temperature': float(min_temp),
            'avg_temperature': float(avg_temp),
            'voltage_imbalance': float(voltage_imbalance),
            'temperature_grid': temp_grid.tolist(),
            'fan_speed': float(fan_speed * 100),  # Convert to percentage
            'pump_level': float(pump_level * 100),  # Convert to percentage
            'risk_level': risk_level,
            'health_status': health_status,
            'time_elapsed': float(self.time_elapsed),
            'pack_voltage': float(self.get_pack_voltage()),
            'avg_current': float(np.mean(currents) if currents else 0),
            'soc_grid': self.get_soc_grid().tolist(),
            'voltage_grid': self.get_voltage_grid().tolist(),
            'temperature_trend': float(temp_trend)
        }
        
        # Ensure all numpy types are converted to native Python types for JSON serialization
        for key, value in state.items():
            if isinstance(value, np.generic):
                state[key] = value.item()
        
        # Store last state for reference
        self.last_state = state
        
        return state

    def train_pinn_models(self, num_samples: int = 1000, epochs: int = 100):
        """Train the PINN models for temperature prediction."""
        if not self.use_pinn:
            print("PINN is disabled. Enable with use_pinn=True")
            return
            
        print("Generating training data...")
        X, y = self._generate_training_data(num_samples)
        
        # Split into training and validation sets
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train each cell's PINN model
        for cell_id, pinn_model in self.pinn_models.items():
            print(f"Training PINN model for cell {cell_id}...")
            pinn_model.train(X_train, y_train, X_val, y_val, epochs=epochs)
            print(f"Cell {cell_id} training complete.")

    def _generate_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for PINN models."""
        X = []
        y = []
        
        for _ in range(num_samples):
            # Generate random but realistic input values
            current_temp = np.random.uniform(20, 50)  # 20-50°C
            current = np.random.uniform(0, 10)       # 0-10A
            soc = np.random.uniform(0.2, 0.8)        # 20-80% SOC
            ambient_temp = np.random.uniform(15, 35)  # 15-35°C
            
            # Simulate temperature changes (simplified physics model)
            temp_10s = current_temp + (current * 0.1) * 10 + np.random.normal(0, 0.1)
            temp_30s = current_temp + (current * 0.1) * 30 + np.random.normal(0, 0.3)
            temp_60s = current_temp + (current * 0.1) * 60 + np.random.normal(0, 0.5)
            
            # Add cooling effect based on ambient temperature
            cooling_effect = (current_temp - ambient_temp) * 0.01
            temp_10s -= cooling_effect * 10
            temp_30s -= cooling_effect * 30
            temp_60s -= cooling_effect * 60
            
            # Clip to reasonable temperature range
            temp_10s = np.clip(temp_10s, 15, 80)
            temp_30s = np.clip(temp_30s, 15, 80)
            temp_60s = np.clip(temp_60s, 15, 80)
            
            X.append([current_temp, current, soc, ambient_temp])
            y.append([temp_10s, temp_30s, temp_60s])
        
        return np.array(X), np.array(y)