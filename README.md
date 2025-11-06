📄 Abstract

With the increasing prevalence of mental health issues, there is a growing need for accessible and effective support tools. Chatbots powered by artificial intelligence offer a promising avenue for providing mental health support in a convenient and non-judgmental manner.

This project explores the development of a mental health chatbot that combines conversational AI with a recommendation system to provide personalized support and guidance.

The chatbot uses Natural Language Processing (NLP) to understand user inputs, identify patterns, and provide relevant responses based on a curated knowledge base of mental health information. Additionally, a recommendation mechanism suggests coping strategies, helpful activities, or supportive responses depending on the user’s emotional state.

This personalized approach aims to enhance the effectiveness of the chatbot in addressing individual mental health concerns and promoting overall well-being.

⚠️ Disclaimer:
This chatbot uses a pretrained NLP model trained on curated conversational datasets related to mental health.
It is designed only for informational and emotional support, not as a replacement for therapy, diagnosis, or medical advice.
Mental health is a sensitive and critical topic — if you are in crisis, please seek help from a licensed professional or emergency service.

🧩 Key Features

 Conversational AI: Understands and responds to user queries about emotions, stress, anxiety, and motivation.

 Pretrained Model: Built using Keras and TensorFlow to ensure consistent and safe responses.

 Intents-Based Learning: Uses an intents.json file containing categorized responses and user patterns.

 NLP Processing: Uses NLTK for tokenization and lemmatization.

 Web Deployment: Hosted using Flask and deployed live on Render.

 Ethical AI: Provides empathetic and supportive responses with sensitivity toward mental health.

⚙️ Tech Stack
| **Component**         | **Technology Used**                                  |
| --------------------- | ---------------------------------------------------- |
| **Frontend**          | HTML, CSS, JavaScript                                |
| **Backend Framework** | Flask                                                |
| **Machine Learning**  | TensorFlow, Keras                                    |
| **NLP Toolkit**       | NLTK                                                 |
| **Deployment**        | Render                                               |
| **Data Format**       | intents.json (custom-trained conversational dataset) |

🧠 How It Works

User Interaction: The user enters a message through the chat interface.

Message Processing: The Flask backend receives the message through the /get?msg= API endpoint.

NLP Pipeline: The message is processed through tokenization, lemmatization, and bag-of-words encoding.

Intent Prediction: The pretrained Keras model predicts the most likely intent.

Response Generation: Based on predicted intent, a response is selected from intents.json.

Recommendation Layer (Optional): Suggests helpful activities, coping strategies, or motivational prompts.
