from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
def init_model(x_train, actions, model_type, num_heads, num_layers):
    model = None
    if model_type =="LSTM":
        model = Sequential([
            LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
            Dense(32, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])
    elif model_type == "Transformer":
        model = TransformerModel(units=64, d_model=x_train.shape[-1], num_heads=num_heads, num_layers=num_layers,
                                 dropout=0.1)
        lstm = Bidirectional(LSTM(64, activation='relu'))(model.output)
        dense1 = Dense(32, activation='relu')(lstm)
        outputs = Dense(len(actions), activation='softmax')(dense1)
        model = Model(inputs=model.input, outputs=outputs)
    return model
def TransformerModel(units, d_model, num_heads, num_layers, dropout):
    inputs = Input(shape=(None, d_model))
    x = inputs

    for _ in range(num_layers):
        attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attention = Dropout(dropout)(attention)
        attention = LayerNormalization(epsilon=1e-6)(x + attention)

        outputs = Dense(units, activation='relu')(attention)
        outputs = Dense(d_model)(outputs)
        outputs = Dropout(dropout)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

        x = outputs

    model = Model(inputs=inputs, outputs=outputs)
    return model

