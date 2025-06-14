import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if ';' in line:
                parts = line.rsplit(';', 1)
                if len(parts) == 2:
                    text, label = parts
                    data.append((text, label))
    return pd.DataFrame(data, columns=['text', 'emotion'])

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return " ".join(tokens)

def train_and_save_model():
    # Load data
    train_df = load_data('train.txt')
    train_df['clean_text'] = train_df['text'].apply(preprocess)
    
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['emotion']
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, vectorizer

def load_model():
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model, vectorizer

def get_suggestions(emotion):
    suggestions = {
        'sadness': [
            "🌈 Keep your face to the sunshine and you cannot see a shadow. – Helen Keller",
            "🎵 Song: 'Fix You' by Coldplay",
            "😂 Joke: Why don't scientists trust atoms? Because they make up everything!"
        ],
        'joy': [
            "💡 Keep smiling, because life is a beautiful thing. – Marilyn Monroe",
            "🎵 Song: 'Happy' by Pharrell Williams",
            "😂 Joke: I told my computer I needed a break, and it said: 'No problem, I'll go to sleep!'"
        ],
        'anger': [
            "🧘‍♂️ For every minute you are angry, you lose sixty seconds of happiness. – Emerson",
            "🎵 Song: 'Let It Go' from Frozen",
            "😂 Joke: I'm on a seafood diet. I see food and I eat it!"
        ],
        'love': [
            "❤️ Love recognizes no barriers. – Maya Angelou",
            "🎵 Song: 'Perfect' by Ed Sheeran",
            "😂 Joke: Are you made of copper and tellurium? Because you're Cu-Te."
        ],
        'fear': [
            "💪 Don't let your fear decide your future. – Shalane Flanagan",
            "🎵 Song: 'Brave' by Sara Bareilles",
            "😂 Joke: What did one wall say to the other wall? 'I'll meet you at the corner!'"
        ]
    }
    return random.choice(suggestions.get(emotion, ["No suggestion available for this emotion."]))