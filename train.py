import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower, remove duplicates
words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to np.array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data created")

# --- Optimized Model ---
model = Sequential()
model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))  # Smaller layer
model.add(Dropout(0.3))  # Less dropout
model.add(Dense(len(train_y[0]), activation='softmax'))  # Output layer

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Save model without optimizer to reduce size
model.save('model.h5', include_optimizer=False)
print("Optimized model created")