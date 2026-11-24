import json
import pickle
import random
import re
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

# Load intents
with open(BASE_DIR / "Include" / "intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# Load words and classes
with open(BASE_DIR / "words.pkl", "rb") as file:
    words = pickle.load(file)

with open(BASE_DIR / "classes.pkl", "rb") as file:
    classes = pickle.load(file)

# Load trained model
model = load_model(BASE_DIR / "chatbot_model.keras")


def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())


def bag_of_words(sentence):
    sentence_words = tokenize(sentence)
    bag = [0] * len(words)

    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1

    return np.array(bag, dtype=np.float32)


def predict_class(sentence):
    bow = bag_of_words(sentence)

    prediction = model.predict(np.array([bow]), verbose=0)[0]

    results = [
        [i, probability]
        for i, probability in enumerate(prediction)
        if probability > 0.25
    ]

    results.sort(key=lambda x: x[1], reverse=True)

    return [
        {"intent": classes[r[0]], "probability": float(r[1])}
        for r in results
    ]


def get_response(intents_list):
    if not intents_list:
        return "Sorry, I don't understand. Please try again."

    tag = intents_list[0]["intent"]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."


print("================================")
print("       WELCOME TO AI CHATBOT")
print("================================")
print("Type quit to exit")

while True:
    message = input("\nYou: ").strip()

    if message.lower() in ["quit", "exit", "stop"]:
        print("Bot: Goodbye! Have a nice day.")
        break

    if not message:
        continue

    result = predict_class(message)
    response = get_response(result)

    print("Bot:", response)
