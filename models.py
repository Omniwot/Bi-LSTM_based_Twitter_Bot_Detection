from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_bidirectional_lstm_model(embedding_matrix):
    """
    Creates and compiles a bidirectional LSTM model for binary classification.
    
    Parameters: embedding_matrix (np.array): Pre-trained embedding matrix (e.g., GloVe embeddings).
        
    Returns: Bidirectional LSTM model.
    """
    # Define the model
    model = Sequential()

    # Embedding layer
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],  # Vocabulary size
        output_dim=embedding_matrix.shape[1],  # Embedding dimension
        weights=[embedding_matrix],  # Pre-trained embeddings
        trainable=False  # Keep embeddings fixed
    ))

    # Add Bidirectional LSTM
    model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.2)))
    # Fully connected layers
    model.add(Dense(32, activation="relu"))  # Dense layer for feature extraction
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def create_hybrid_bilstm_model(embedding_matrix, input_length, numerical_input_dim):
    """
    Creates a hybrid BiLSTM model for text and numerical metadata input.

    Parameters:
        embedding_matrix (np.array): Pre-trained embedding matrix (e.g., GloVe embeddings).
        input_length (int): Length of input sequences (padded sequence length).
        numerical_input_dim (int): Dimension of the numerical input features.

    Returns:  Hybrid BiLSTM model.
    """
    # Text input
    text_input = Input(shape=(input_length,), name="Text_Input")
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False
    )(text_input)
    lstm_output = Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.2))(embedding_layer)

    # Numerical input
    numerical_input = Input(shape=(numerical_input_dim,), name="Numerical_Input")
    dense_numerical = Dense(8, activation="relu")(numerical_input) # Dense layer for feature extraction from numerical input

    # Combine text and numerical features
    combined = Concatenate()([lstm_output, dense_numerical])

    # Fully connected layers
    dense_combined = Dense(32, activation="relu")(combined) # Dense layer for feature extraction from combined output
    dense_combined = Dropout(0.5)(dense_combined)
    output = Dense(1, activation="sigmoid", name="Output_Layer")(dense_combined) # Output layer for binary classification

    # Define and compile the model
    model = Model(inputs=[text_input, numerical_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

# Define the hybrid model function
def create_hybrid_bilstm_model_with_lr(embedding_matrix, input_length, numerical_input_dim, learning_rate):
    """
    Creates a hybrid BiLSTM model for text and numerical metadata input.

    Parameters:
        embedding_matrix : Pre-trained embedding matrix (e.g., GloVe embeddings).
        input_length : Length of input sequences (padded sequence length).
        numerical_input_dim : Dimension of the numerical input features.
        learning_rate: Learning rate of the Adam optimizer.

    Returns: Hybrid BiLSTM model.
    """
    # Text input
    text_input = Input(shape=(input_length,), name="Text_Input")
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False
        )(text_input)

    # BiLSTM layer
    lstm_output = Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.2))(embedding_layer)

    # Numerical input
    numerical_input = Input(shape=(numerical_input_dim,), name="Numerical_Input")
    dense_numerical = Dense(8, activation="relu")(numerical_input)

    # Combine text and numerical features
    combined = Concatenate()([lstm_output, dense_numerical])

    # Fully connected layers
    dense_combined = Dense(32, activation="relu")(combined)
    dense_combined = Dropout(0.5)(dense_combined)
    output = Dense(1, activation="sigmoid")(dense_combined)

    # Define and compile the model
    model = Model(inputs=[text_input, numerical_input], outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model