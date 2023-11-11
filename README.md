# Twitter Sentiment Analysis

## Overview
This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) and machine learning models. The goal is to classify tweets into categories such as Positive, Negative, Neutral, or Irrelevant.

## Features
- **Data Preprocessing:** Clean and preprocess tweet data, removing special characters and stopwords.
- **Text Vectorization:** Convert text data into numerical features using TF-IDF.
- **Model Selection:** Explore various classification algorithms, including Naive Bayes and XGBoost.
- **Web Interface:** Create a simple Flask app for users to input text for sentiment analysis.

## Project Structure
- **Data:** Twitter dataset with labeled sentiments for training and validation.
- **Code:** Jupyter notebooks for data analysis, model training, and Flask app for deployment.
- **Models:** Pre-trained models saved for easy deployment.

## Usage
1. **Clone the repository:** `git clone https://github.com/yourusername/sentiment-analysis.git`
2. **Navigate to the project directory:** `cd sentiment-analysis`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Run the Flask app:** `python sentiment_analysis_app.py`
5. **Access the web interface:** [http://127.0.0.1:5000](http://127.0.0.1:5000) to input text for sentiment analysis.

## Dependencies
- Flask
- scikit-learn
- xgboost
- nltk

