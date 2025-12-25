import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class RiskLSTM:
    def __init__(self, input_dim=4, seq_len=60):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(32, input_shape=(self.seq_len, self.input_dim), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))  # Risk score between 0 and 1
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        callbacks = []
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def predict(self, X_seq):
        # X_seq shape: (batch, seq_len, input_dim)
        return self.model.predict(X_seq, verbose=0)

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath, input_dim=4, seq_len=60):
        from tensorflow.keras.models import load_model
        risk_lstm = cls(input_dim=input_dim, seq_len=seq_len)
        risk_lstm.model = load_model(filepath)
        return risk_lstm
