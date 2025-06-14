from flask import Flask, render_template, request, redirect, url_for, session
from utils import load_model, preprocess, get_suggestions
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load model and vectorizer
try:
    model, vectorizer = load_model()
except:
    from utils import train_and_save_model
    model, vectorizer = train_and_save_model()

@app.before_request
def before_request():
    if 'history' not in session:
        session['history'] = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        return redirect(url_for('predict', text=text))
    return render_template('index.html')

@app.route('/predict')
def predict():
    text = request.args.get('text', '')
    if not text:
        return redirect(url_for('index'))
    
    # Preprocess and predict
    cleaned_text = preprocess(text)
    vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    # Get top 3 emotions with probabilities
    emotions = model.classes_
    emotion_probs = list(zip(emotions, proba))
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    top_emotions = emotion_probs[:3]
    
    # Get suggestion
    suggestion = get_suggestions(prediction)
    
    # Add to history
    analysis = {
        'text': text,
        'emotion': prediction,
        'suggestion': suggestion,
        'top_emotions': top_emotions,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    session['history'].append(analysis)
    session.modified = True
    
    return render_template('results.html', 
                         text=text,
                         emotion=prediction,
                         suggestion=suggestion,
                         top_emotions=top_emotions)

@app.route('/history')
def history():
    return render_template('history.html', history=session.get('history', []))

@app.route('/clear_history')
def clear_history():
    session['history'] = []
    session.modified = True
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)