# chatbot

import random
import json
import torch
import pandas as pd
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path to CSV file
csv_file_path = 'intents.csv'

with open(csv_file_path, 'r') as file:
    # Use pandas read_csv to load the CSV file into a DataFrame
    intents = pd.read_csv(file)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_intent_and_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    return tag, prob.item()

def generate_response(intent):
    for index, row in intents.iterrows():
        if row['tag'] == intent:
            responses = row['responses']
            return random.choice(eval(responses))

    return "I'm not sure how to respond to that."


def get_response(msg):
    intent, confidence = get_intent_and_response(msg)

    if confidence > 0.75:
        response = generate_response(intent)
    else:
        response = "I do not understand, may you please rephrase your question."

    return response