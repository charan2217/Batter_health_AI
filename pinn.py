# models/pinn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

class PINN:
    def __init__(self, input_dim=4, output_dim=3):
        # Initialize TensorFlow session if not already done
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = self._build_model(input_dim, output_dim)
        # Add scaling parameters
        self.X_min = np.array([20, 0, 0.2, 15])  # min values for [temp, current, soc, ambient_temp]
        self.X_max = np.array([50, 10, 0.8, 35])  # max values for [temp, current, soc, ambient_temp]
        self.y_min = 15  # min temperature in °C
        self.y_max = 80  # max temperature in °C

    def _build_model(self, input_dim, output_dim):
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='tanh')(inputs)
        x = Dense(64, activation='tanh')(x)
        outputs = Dense(output_dim, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def _scale_X(self, X):
        """Scale input features to [0, 1] range"""
        return (X - self.X_min) / (self.X_max - self.X_min)

    def _scale_y(self, y):
        """Scale output temperatures to [0, 1] range"""
        return (y - self.y_min) / (self.y_max - self.y_min)

    def _inverse_scale_y(self, y_scaled):
        """Convert scaled predictions back to °C"""
        return y_scaled * (self.y_max - self.y_min) + self.y_min

    def train(self, X_train, y_train, X_val, y_val, epochs=300, batch_size=32):
        # Scale the data
        X_train_scaled = self._scale_X(X_train)
        y_train_scaled = self._scale_y(y_train)
        X_val_scaled = self._scale_X(X_val)
        y_val_scaled = self._scale_y(y_val)
        
        # Train the model within the same graph
        with self.graph.as_default():
            history = self.model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        return history

    def predict(self, X):
        """Make predictions and return them in °C"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self._scale_X(X)
        with self.graph.as_default():
            y_scaled = self.model.predict(X_scaled, verbose=0)
        return self._inverse_scale_y(y_scaled)

    def save(self, filepath):
        with self.graph.as_default():
            self.model.save(filepath)
        np.savez(
            f"{filepath}_scaling.npz",
            X_min=self.X_min,
            X_max=self.X_max,
            y_min=self.y_min,
            y_max=self.y_max
        )

    @classmethod
    def load(cls, filepath):
        # Create a new instance
        instance = cls()
        # Load the model within the graph context
        with instance.graph.as_default():
            instance.model = load_model(filepath, compile=True)
        scaling = np.load(f"{filepath}_scaling.npz")
        instance.X_min = scaling['X_min']
        instance.X_max = scaling['X_max']
        instance.y_min = scaling['y_min']
        instance.y_max = scaling['y_max']
        return instance
        return pinn