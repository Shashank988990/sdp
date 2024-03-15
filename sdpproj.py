from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'sdp12345'  # Change this to a secure secret key

# Load the dataset
import os
file_path = os.path.join(os.getcwd(), 'tweet_dataset.csv')
df = pd.read_csv(file_path, encoding='latin1')

# Convert tweet texts to lowercase
df['text'] = df['text'].str.lower()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on the dataset
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

@app.route('/')
def index():
    if 'email' in session:
        return render_template('index.html', email=session['email'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        if email.endswith('@gmail.com'):
            session['email'] = email
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error_message='Please enter a valid Gmail address.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        return redirect(url_for('login'))

    tweet_text = request.form.get('tweet_text')

    # Convert the input tweet text to lowercase
    tweet_text = tweet_text.lower()

    # Remove the '#' symbol if present
    tweet_text = tweet_text.replace('#', '')

    # Initialize image_path with a default value
    image_path = None

    # Initialize tweet_info with a default value
    tweet_info = None

    # Transform the input tweet text into a TF-IDF vector
    input_tweet_vector = tfidf_vectorizer.transform([tweet_text])

    # Calculate cosine similarity between the input tweet and all tweets in the dataset
    similarities = cosine_similarity(input_tweet_vector, tfidf_matrix)

    # Find the indices of matching tweets in the dataset
    matching_indices = similarities.argsort()[0][::-1]

    # Calculate the average likes for matching tweets
    total_likes = 0
    count = 0
    for idx in matching_indices:
        if tweet_text in df.loc[idx, 'text']:  # Check if the input tweet text exactly matches
            total_likes += df.loc[idx, 'likes']
            count += 1
            # Get image path and tweet info for the input tweet
            image_path = df.loc[idx, 'image']
            tweet_info = df.loc[idx, 'info']
        if count >= 5:
            break

    # Calculate the average likes for matching tweets
    predicted_likes = total_likes / count if count > 0 else 0

    return render_template('index.html', email=session['email'], tweet_text=tweet_text,
                           predicted_likes=predicted_likes, image_path=image_path, tweet_info=tweet_info)

if __name__ == '__main__':
    app.run(debug=True)
