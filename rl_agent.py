import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model
import time

@dataclass
class ActionLog:
    time: float
    max_temp: float
    action: Tuple[float, float, float]  # (fan_speed, pump_level, charge_rate)
    action_desc: str
    cooling_effect: float
    risk_score: float
    confidence: float
    predicted_temps: Tuple[float, float, float]  # 10s, 30s, 60s

class RLAgent:
    def __init__(self, temp_threshold: float = 35.0, learning_rate: float = 0.0003):
        self.temp_threshold = temp_threshold
        self.risk_threshold = 0.3  # Lower risk threshold for earlier intervention
        self.learning_rate = learning_rate
        self.action_history: List[ActionLog] = []
        self.temperature_history = []
        self.risk_history = []
        self.manual_cooling = {'is_active': False, 'fan_speed': 0.0, 'pump_level': 0.0}
        
        # Define action space
        self.action_space = self._create_action_space()
        self.state_dim = 8  # [max_temp, avg_temp, min_temp, temp_trend, 
                           #  risk_score, risk_confidence, soc, time_since_last_action]
        self.action_dim = 3  # fan_speed, pump_level, charge_rate
        
        # Build actor-critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Training parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005  # target network update rate
        self.entropy_coef = 0.01  # encourages exploration
        
        # Target networks
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        self.update_target_networks(tau=1.0)  # hard update

    def _build_actor(self):
        """Build actor network that outputs mean and std for each action dimension"""
        inputs = layers.Input(shape=(self.state_dim,))
        
        # Feature extraction
        x = layers.Dense(256, activation='swish')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation='swish')(x)
        x = layers.LayerNormalization()(x)
        
        # Output mean and log_std as separate outputs
        mean = layers.Dense(self.action_dim, activation='sigmoid', name='mean')(x)
        
        # Create log_std as a non-trainable variable wrapped in a Lambda layer
        log_std = layers.Lambda(
            lambda x: tf.ones((self.action_dim,), dtype=tf.float32) * -0.5,  # Initialize to -0.5
            name='log_std'
        )(x)
        
        # Create model with two outputs
        model = Model(inputs=inputs, outputs=[mean, log_std])
        return model

    def _build_critic(self):
        """Build critic network that estimates state value"""
        inputs = layers.Input(shape=(self.state_dim,))
        
        x = layers.Dense(256, activation='swish')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation='swish')(x)
        x = layers.LayerNormalization()(x)
        value = layers.Dense(1, activation=None)(x)  # Q-value
        
        return Model(inputs=inputs, outputs=value)

    def _create_action_space(self):
        """Create a discrete set of possible actions"""
        # [fan_speed, pump_level, charge_rate]
        return [
            [0.2, 0.0, 1.0],   # Minimal cooling, full charge
            [0.5, 0.0, 0.8],   # Moderate cooling, slightly reduced charge
            [0.8, 0.3, 0.6],   # Active cooling, reduced charge
            [1.0, 0.6, 0.3],   # High cooling, minimal charge
            [1.0, 1.0, 0.0]    # Maximum cooling, stop charging
        ]

    def get_action(self, state: Dict[str, Any], evaluate: bool = False) -> Tuple[np.ndarray, float]:
        """Select action using the current policy"""
        # Prepare state vector
        state_vec = self._process_state(state)
        
        # Get action distribution
        mean, log_std = self.actor(np.expand_dims(state_vec, axis=0))
        
        if evaluate:
            # Use mean action during evaluation
            action = mean[0].numpy()
        else:
            # Sample from normal distribution during training
            std = tf.exp(log_std)
            noise = tf.random.normal(shape=mean.shape, mean=0.0, stddev=1.0)
            action = mean + std * noise
            action = action[0].numpy()  # Remove batch dimension
            
            # Clip to valid range [0, 1]
            action = np.clip(action, 0, 1)
            
            # Ensure charge rate is 0 when temperature is critical
            if state.get('max_temperature', 0) > 45 or state.get('risk_score', 0) > 0.7:
                action[2] = 0.0  # Stop charging
                
            # Ensure pump is on if fan is at max
            if action[0] > 0.8 and action[1] < 0.3:
                action[1] = 0.3
                
        return action, 0.0  # Return action and log_prob (0 for deterministic)

    def _process_state(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to numpy array"""
        max_temp = state.get('max_temperature', 25.0)
        avg_temp = state.get('avg_temperature', max_temp)
        min_temp = state.get('min_temperature', max_temp)
        temp_trend = state.get('temperature_trend', 0.0)
        risk_score = state.get('risk_score', 0.0)
        risk_confidence = state.get('risk_confidence', 0.0)
        soc = state.get('avg_soc', 0.5)
        
        # Time since last action (normalized to [0,1])
        time_since_last = 0.0
        if self.action_history:
            time_since_last = min(1.0, (time.time() - self.action_history[-1].time) / 60.0)  # Normalize to 1 minute
            
        return np.array([
            max_temp / 80.0,  # Normalize by max expected temp
            avg_temp / 80.0,
            min_temp / 80.0,
            np.tanh(temp_trend),  # Bound trend to [-1, 1]
            risk_score,
            risk_confidence,
            soc,
            time_since_last
        ], dtype=np.float32)

    def update(self, state: Dict[str, Any], action: np.ndarray, 
               reward: float, next_state: Dict[str, Any], done: bool):
        """Update the agent's policy using PPO"""
        state_vec = self._process_state(state)
        next_state_vec = self._process_state(next_state)
        
        with tf.GradientTape() as tape:
            # Get current action probabilities
            mean, log_std = self.actor(np.expand_dims(state_vec, axis=0))
            std = tf.exp(log_std)
            noise = tf.random.normal(shape=mean.shape, mean=0.0, stddev=1.0)
            action_dist = mean + std * noise
            
            # Get state values
            value = self.critic(np.expand_dims(state_vec, axis=0))
            next_value = self.target_critic(np.expand_dims(next_state_vec, axis=0))
            
            # Calculate advantage and returns
            returns = reward + (1 - done) * self.gamma * next_value
            advantage = returns - value
            
            # Calculate policy loss
            ratio = tf.exp(tf.reduce_sum(tf.math.log(action_dist) - tf.math.log(std), axis=1))
            policy_loss = -tf.minimum(
                ratio * advantage,
                tf.clip_by_value(ratio, 1-0.2, 1+0.2) * advantage
            )
            
            # Value loss
            value_loss = tf.keras.losses.MeanSquaredError()(returns, value)
            
            # Entropy bonus for exploration (approximate for normal distribution)
            entropy = tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * np.e * tf.square(std)))
            
            # Total loss
            total_loss = -policy_loss + 0.5 * value_loss - 0.01 * self.entropy_coef * entropy
            
        # Compute gradients and update
        grads = tape.gradient(total_loss, self.actor.trainable_variables + 
                             self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(
            grads, 
            self.actor.trainable_variables + self.critic.trainable_variables
        ))
        
        # Update target networks
        self.update_target_networks()
        
        return {
            'policy_loss': float(tf.reduce_mean(policy_loss)),
            'value_loss': float(tf.reduce_mean(value_loss)),
            'entropy': float(entropy)
        }

    def update_target_networks(self, tau: float = None):
        """Update target networks using polyak averaging"""
        tau = tau or self.tau
        for target, source in zip(self.target_actor.variables, self.actor.variables):
            target.assign(tau * source + (1.0 - tau) * target)
            
        for target, source in zip(self.target_critic.variables, self.critic.variables):
            target.assign(tau * source + (1.0 - tau) * target)
    
    def act(self, battery_pack, time_elapsed: float, state: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float]:
        """Take action based on current state"""
        if state is None:
            state = battery_pack.get_state()
            
        # Get current temperatures and risk
        max_temp = state['max_temperature']
        risk_score = state.get('risk_score', 0.0)
        risk_confidence = state.get('risk_confidence', 0.0)
        
        # Store state for training
        self.temperature_history.append(max_temp)
        self.risk_history.append(risk_score)
        
        # Get action from policy
        action, _ = self.get_action(state, evaluate=False)
        fan_speed, pump_level, charge_rate = action
        
        # Log the action
        action_desc = self._get_action_description(fan_speed, pump_level, charge_rate)
        cooling_effect = self._calculate_cooling_effect(fan_speed, pump_level, max_temp, 
                                                      state.get('ambient_temp', 25.0))
        
        # Get temperature predictions if available
        predicted_temps = state.get('predicted_temps', (max_temp, max_temp, max_temp))
        
        # Log the action
        action_tuple = (float(fan_speed), float(pump_level), float(charge_rate))
        self.action_history.append(ActionLog(
            time=time.time(),
            max_temp=max_temp,
            action=action_tuple,
            action_desc=action_desc,
            cooling_effect=cooling_effect,
            risk_score=risk_score,
            confidence=risk_confidence,
            predicted_temps=predicted_temps
        ))
        
        # Return the action tuple
        return action_tuple

    def _get_action_description(self, fan_speed: float, pump_level: float, charge_rate: float) -> str:
        """Generate human-readable action description"""
        if fan_speed < 0.3 and pump_level < 0.1:
            return "Minimal cooling"
        elif fan_speed < 0.6 and pump_level < 0.3:
            return "Moderate cooling"
        elif fan_speed < 0.9 or pump_level < 0.7:
            return "Active cooling"
        else:
            return "Maximum cooling"
    
    def _calculate_cooling_effect(self, fan_speed: float, pump_level: float, 
                                current_temp: float, ambient_temp: float) -> float:
        """Estimate cooling effect in °C/s"""
        # Use manual cooling settings if active
        if hasattr(self, 'manual_cooling') and self.manual_cooling.get('is_active', False):
            fan_speed = self.manual_cooling['fan_speed']
            pump_level = self.manual_cooling['pump_level']
        
        # Base cooling from fans (convection)
        fan_cooling = 0.1 * fan_speed * (current_temp - ambient_temp)
        
        # Additional cooling from liquid (more effective at higher temperatures)
        liquid_cooling = 0.15 * pump_level * (current_temp - ambient_temp) * (1 + 0.01 * (current_temp - 30))
        
        return fan_cooling + liquid_cooling
    
    def set_cooling(self, fan_speed: float, pump_level: float) -> None:
        """
        Manually set cooling parameters (for dashboard control)
        
        Args:
            fan_speed: Fan speed in range [0, 1]
            pump_level: Pump level in range [0, 1]
        """
        self.manual_cooling = {
            'fan_speed': np.clip(fan_speed, 0.0, 1.0),
            'pump_level': np.clip(pump_level, 0.0, 1.0),
            'is_active': True
        }
        
    def save(self, filepath: str):
        """Save model weights to disk"""
        self.actor.save_weights(f"{filepath}_actor.h5")
        self.critic.save_weights(f"{filepath}_critic.h5")
        self.target_actor.save_weights(f"{filepath}_target_actor.h5")
        self.target_critic.save_weights(f"{filepath}_target_critic.h5")
        
    @classmethod
    def load(cls, filepath: str, **kwargs):
        """Load model weights from disk"""
        agent = cls(**kwargs)
        agent.actor.load_weights(f"{filepath}_actor.h5")
        agent.critic.load_weights(f"{filepath}_critic.h5")
        agent.target_actor.load_weights(f"{filepath}_target_actor.h5")
        agent.target_critic.load_weights(f"{filepath}_target_critic.h5")
        return agent