from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import joblib

# Load your trained XGBoost model
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.load_model('model.xgb')

# Load your trained TF-IDF vectorizer
tfidf_vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

app = Flask("Twitter Sentiment analysis")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        user_input_processed = preprocess_text(user_input)
        user_input_tfidf = tfidf_vectorizer.transform([user_input_processed])

        # Load your trained LabelEncoder
        label_encoder = joblib.load('label_encoder.pkl')

        prediction_encoded = xgb_classifier.predict(user_input_tfidf)[0]
        predicted_type = label_encoder.inverse_transform([prediction_encoded])[0]
        
        print(f'Predicted Type: {predicted_type}')

        return render_template('index.html', prediction=f'The predicted type is: {predicted_type}')

if __name__ == '__main__':
    app.run(debug=True)
