import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request
import random

# Load NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load static files (JSON, pickle) at startup
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Model lazy loading
model = None
def get_model():
    global model
    if model is None:
        model = load_model('model.h5')
    return model

# --- Utility functions ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

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

def predict_class(sentence):
    model = get_model()
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list if return_list else [{"intent": "no-response", "probability": "0"}]

def getResponse(ints):
    tag = ints[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm here to listen. Could you tell me a bit more about how you're feeling?"

def chatbot_response(msg):
    ints = predict_class(msg)
    res = getResponse(ints)
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

if __name__ == "__main__":
    app.run()
