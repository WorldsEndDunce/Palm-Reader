from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
def init_model(x_train, actions, model_type, nhead):
    model = None
    if model_type =="LSTM":
        model = Sequential([
            LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
            Dense(32, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])
    elif model_type == "Transformer":
        # Define the model
        inputs = Input(shape=x_train.shape[1:3])
        transformer = TransformerLayer(units=64, d_model=x_train.shape[-1], num_heads=nhead, dropout=0.1)(inputs)
        lstm = Bidirectional(LSTM(64, activation='relu'))(transformer)
        dense1 = Dense(32, activation='relu')(lstm)
        outputs = Dense(len(actions), activation='softmax')(dense1)
        model = Model(inputs=inputs, outputs=outputs)
    return model
def TransformerLayer(units, d_model, num_heads, dropout):
    inputs = Input(shape=(None, d_model))
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = Dense(units, activation='relu')(attention)
    outputs = Dense(d_model)(outputs)
    outputs = Dropout(dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

    return Model(inputs=inputs, outputs=outputs)

