import nltk
import numpy as np
from nltk.corpus import treebank
from nltk.tag import hmm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
import tensorflow as tf

nltk.download('treebank')

# 1. Load and Prepare Data
tagged_sentences = treebank.tagged_sents()
train_data, test_data = train_test_split(tagged_sentences, test_size=0.2, random_state=42)

# Prepare data for HMM
hmm_train_data = train_data
hmm_test_data = test_data

# Prepare data for Neural Network
def untag(sentences):
    words, tags = [], []
    for sent in sentences:
        words.append([word for word, tag in sent])
        tags.append([tag for word, tag in sent])
    return words, tags

words_train, tags_train = untag(train_data)
words_test, tags_test = untag(test_data)

# Flatten the word/tag lists for tokenization
flat_words_train = [word for sentence in words_train for word in sentence]
flat_words_test = [word for sentence in words_test for word in sentence]
flat_tags_train = [tag for sentence in tags_train for tag in sentence]
flat_tags_test = [tag for sentence in tags_test for tag in sentence]

# 2. HMM Model
hmm_tagger = hmm.HiddenMarkovModelTagger.train(hmm_train_data)

def evaluate_hmm(test_data, tagger):
    total_correct = 0
    total_words = 0
    for sentence in test_data:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]
        predicted_tags = [tag for word, tag in tagger.tag(words)]
        for pred, true in zip(predicted_tags, true_tags):
            if pred == true:
                total_correct += 1
            total_words += 1
    return total_correct / total_words

hmm_accuracy = evaluate_hmm(hmm_test_data, hmm_tagger)
print(f"HMM Accuracy: {hmm_accuracy:.4f}")

# 3. Neural Network (LSTM)

# Tokenize words
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(flat_words_train)
X_train = tokenizer.texts_to_sequences(words_train)
X_test = tokenizer.texts_to_sequences(words_test)

# Pad sequences
max_len = max(len(seq) for seq in X_train)
X_train_padded = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test, maxlen=max_len, padding='post')

# Tag to index mapping
tag_set = sorted(set(flat_tags_train))
tag2idx = {tag: i for i, tag in enumerate(tag_set)}
idx2tag = {i: tag for tag, i in tag2idx.items()}

# Convert tags to indexed and pad
y_train = [[tag2idx[tag] for tag in seq] for seq in tags_train]
y_test = [[tag2idx.get(tag, 0) for tag in seq] for seq in tags_test]

y_train_padded = pad_sequences(y_train, maxlen=max_len, padding='post')
y_test_padded = pad_sequences(y_test, maxlen=max_len, padding='post')

# One-hot encode tags
y_train_onehot = to_categorical(y_train_padded, num_classes=len(tag_set))
y_test_onehot = to_categorical(y_test_padded, num_classes=len(tag_set))

# Build model
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=max_len),
    LSTM(64, return_sequences=True),
    Dense(len(tag_set), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
model.fit(X_train_padded, y_train_onehot, batch_size=128, epochs=5, validation_split=0.1)

# Evaluate
loss, nn_accuracy = model.evaluate(X_test_padded, y_test_onehot, verbose=0)
print(f"Neural Network (LSTM) Accuracy: {nn_accuracy:.4f}")
