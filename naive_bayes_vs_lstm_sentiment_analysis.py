import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore Z
from tensorflow.keras.utils import to_categorical  # type: ignore
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf



# Load the IMDb dataset from TensorFlow Datasets
(ds_train, ds_test), ds_info = tfds.load(
    'imdb_reviews',
    split=('train', 'test'),
    as_supervised=True,  # Get data as (text, label) tuples
    with_info=True,
)

# Convert TensorFlow Datasets to lists of text and labels
texts_train = []
labels_train = []
for text, label in tfds.as_numpy(ds_train):
    texts_train.append(text.decode('utf-8'))  # Decode the byte strings
    labels_train.append(label)

texts_test = []
labels_test = []
for text, label in tfds.as_numpy(ds_test):
    texts_test.append(text.decode('utf-8'))
    labels_test.append(label)

# Convert labels to numpy arrays
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)

# Define number of classes.  IMDB is binary, so 2.
num_classes = 2

# 2. Preprocess the Data
# --- Text Vectorization (TF-IDF for Naive Bayes, Tokenization for Neural Network) ---
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(texts_train)
X_test_tfidf = tfidf_vectorizer.transform(texts_test)

# Tokenization for Neural Network
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts_train)
X_train_tokens = tokenizer.texts_to_sequences(texts_train)
X_test_tokens = tokenizer.texts_to_sequences(texts_test)

# Pad sequences to ensure equal length
max_length = 200  # Increased max length, adjust as needed
X_train_padded = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

# One-hot encode the labels for the neural network
y_train_onehot = to_categorical(labels_train, num_classes=num_classes)
y_test_onehot = to_categorical(labels_test, num_classes=num_classes)


# 3. Train the Models

# --- Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, labels_train)

# --- Neural Network (LSTM) ---
# Define the LSTM model
nn_model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=max_length), # Adjusted embedding dim
    LSTM(64),  # Adjusted LSTM units
    Dense(num_classes, activation='softmax')
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.summary()

# Train the model
history = nn_model.fit(X_train_padded, y_train_onehot, epochs=10, batch_size=128, verbose=1, validation_split=0.1) #Added batch size and reduced epochs


# 4. Evaluate the Models

# --- Naive Bayes Evaluation ---
nb_predictions = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(labels_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)
print(classification_report(labels_test, nb_predictions, target_names=['positive', 'negative']))


# --- Neural Network Evaluation ---
nn_predictions = nn_model.predict(X_test_padded)
nn_predictions_labels = np.argmax(nn_predictions, axis=1)
nn_accuracy = accuracy_score(labels_test, nn_predictions_labels)
print("Neural Network Accuracy:", nn_accuracy)
print(classification_report(labels_test, nn_predictions_labels, target_names=['positive', 'negative']))

# 5. Plotting Learning Curve (Neural Network)
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Learning Curve (LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
