import os
import nltk
import random
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download("stopwords")

# Define the path to your downloaded movie reviews dataset
dataset_directory = "txt_sentoken"

# Function to classify review sentiment
def classify_review(sum_score):
    return "pos" if sum_score > 0 else "neg"

# Function to get sentiment for an adverb
def get_sentiment(adverb):
    synsets = list(swn.senti_synsets(adverb, 'r'))  # 'r' for adverbs
    if not synsets:
        return 0  # If not found by SentiWordNet
    return synsets[0].pos_score() - synsets[0].neg_score()

# Function to find adverbs in a review
def find_adverbs(tagged_review):
    adverbs = [word for word, tag in tagged_review if tag.startswith('RB')]
    return adverbs

# Load the movie reviews data
if not os.path.exists(dataset_directory):
    st.error("Dataset directory not found. Please make sure you provide the correct path.")
else:
    pos_reviews = []
    neg_reviews = []
    pos_directory = os.path.join(dataset_directory, "pos")
    neg_directory = os.path.join(dataset_directory, "neg")

    for filename in os.listdir(pos_directory):
        with open(os.path.join(pos_directory, filename), "r", encoding="utf-8") as file:
            review = file.read()
            pos_reviews.append((review, 'positive'))

    for filename in os.listdir(neg_directory):
        with open(os.path.join(neg_directory, filename), "r", encoding="utf-8") as file:
            review = file.read()
            neg_reviews.append((review, 'negative'))

    all_reviews = pos_reviews + neg_reviews
    random.shuffle(all_reviews)

# Streamlit app
st.title("Movie Review Sentiment Analysis")

# Option to view a sample review
if st.checkbox("Show a sample review"):
    sample_review, sample_label = random.choice(all_reviews)
    st.write(f"**Review:**\n{sample_review}")
    st.write(f"**Label:** {sample_label}")

# Sentiment Analysis
st.header("Sentiment Analysis")

# Identify adverbs and sentiment
tagged_reviews = [(word_tokenize(review), label) for review, label in all_reviews]
tagged_reviews = [(pos_tag(tokens), label) for tokens, label in tagged_reviews]
adverbs_in_reviews = [(find_adverbs(tagged_review), label) for tagged_review, label in tagged_reviews]

# Calculate sentiment scores
sentiments_in_reviews = [(sum(get_sentiment(adverb) for adverb in adverbs), label) for adverbs, label in adverbs_in_reviews]
predicted_labels = [classify_review(score) for score, _ in sentiments_in_reviews]

# Display accuracy
actual_labels = [label for _, label in sentiments_in_reviews]
correctly_classified = sum(1 for predicted, actual in zip(predicted_labels, actual_labels) if predicted == actual)
accuracy = correctly_classified / len(predicted_labels)
st.write(f"Accuracy of classification: {accuracy * 100:.2f}%")

# Machine Learning Classification
st.header("Machine Learning Classification")

X = [sum_ for sum_, _ in sentiments_in_reviews]
y = [label for _, label in sentiments_in_reviews]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Logistic Regression
st.subheader("Logistic Regression")

X_train = [[x] for x in X_train]
X_test = [[x] for x in X_test]

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy*100:.2f}%")

# Random Forest Classifier
st.subheader("Random Forest Classifier")

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

all_reviews = [review[0] for review in all_reviews]
X = vectorizer.fit_transform(all_reviews)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy*100:.2f}%")

st.subheader("Classification Report")
report = classification_report(y_test, y_pred)
st.write(report)



