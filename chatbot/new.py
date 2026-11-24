
import json
import pickle
import random
import re
import numpy as np
import tensorflow as tf
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "Include" / "intents.json"

# Load intents
with open(INTENTS_PATH, "r", encoding="utf-8") as file:
    intents = json.load(file)

words = []
classes = []
documents = []

# Tokenization without NLTK
def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())

# Read patterns from intents.json
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
with open(BASE_DIR / "words.pkl", "wb") as file:
    pickle.dump(words, file)

with open(BASE_DIR / "classes.pkl", "wb") as file:
    pickle.dump(classes, file)

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [0] * len(words)

    for word in document[0]:
        if word in words:
            bag[words.index(word)] = 1

    output_row = output_empty.copy()
    output_row[classes.index(document[1])] = 1

    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training, dtype=np.float32)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Create neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(words),)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(classes), activation="softmax")
])

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    ),
    metrics=["accuracy"]
)

# Train model
print("Training chatbot...")

model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=5,
    verbose=1
)

# Save model
model.save(BASE_DIR / "chatbot_model.keras")

print("\nChatbot training completed successfully!")
print("words.pkl created")
print("classes.pkl created")
print("chatbot_model.keras created")
