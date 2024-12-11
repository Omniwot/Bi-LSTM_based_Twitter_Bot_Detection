import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_twitter_data(df, glove_path="glove.6B.100d.txt", max_length=100):
    """
    Preprocess Twitter data, keeping all features in a DataFrame and separating the target variable.

    Parameters:
        df (pd.DataFrame): The input dataset.
        glove_path (str): Path to GloVe embeddings file.
        max_length (int): Maximum sequence length for padding.

    Returns:
        X (pd.DataFrame): DataFrame with all features, including padded sequences.
        y (np.array): Target labels.
        embedding_matrix (np.array): Pre-trained GloVe embedding matrix.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    import re
    
    # Fill missing values
    df["Tweet_text"] = df["Tweet_text"].fillna("")

    # Clean Tweet_text
    def clean_tweet_text(text):
        text = text.lstrip("b'").rstrip("'")  # Remove leading 'b' and trailing quotes
        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphanumeric characters
        return text.strip()

    df["Cleaned_Tweet"] = df["Tweet_text"].apply(clean_tweet_text)

    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["Cleaned_Tweet"])
    sequences = tokenizer.texts_to_sequences(df["Cleaned_Tweet"])
    word_index = tokenizer.word_index
    X_padded = pad_sequences(sequences, maxlen=max_length, padding="post")

    # Store padded sequences in the DataFrame
    df["Padded_Sequences"] = list(X_padded)

    # Load GloVe embeddings
    embedding_dim = 100
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coeffs

    print(f"Loaded {len(embeddings_index)} GloVe word vectors.")

    # Create embedding matrix
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Separate features and target
    X = df.drop(columns=["Label","Tweet_id","Tweet_created_at"])
    y = df["Label"].to_numpy()

    return X, y, embedding_matrix