import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request
import random

# --- NLTK setup ---
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# --- Load model and data with full paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = load_model(os.path.join(BASE_DIR, 'model.h5'))
    print("Model loaded successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Failed to load model:", e)

try:
    with open(os.path.join(BASE_DIR, 'texts.pkl'), 'rb') as f:
        words = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'labels.pkl'), 'rb') as f:
        classes = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'intents.json'), 'r', encoding='utf-8') as f:
        intents = json.load(f)
    print("Pickle and JSON files loaded successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Failed to load data files:", e)

# --- Utility functions ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list if return_list else [{"intent": "no-response", "probability": "0"}]

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm here to listen. Could you tell me a bit more about how you're feeling?"

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# --- Flask setup ---
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("User input:", userText)
    try:
        response = chatbot_response(userText)
        print("Bot response:", response)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "Error: " + str(e)

# --- Run on Render ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT
    app.run(host="0.0.0.0", port=port)
