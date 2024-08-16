import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import nltk  # Natural Language Toolkit
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing functions


def lower_case(text):
    return text.lower()


def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def tokenize(text):
    return text.split(' ')


def remove_stopword(token):
    stop_words = nltk.corpus.stopwords.words('english')
    return [word for word in token if word not in stop_words]


def stemming(token):
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(word) for word in token]


def preprocess_text(text):
    text = punctuation_removal(lower_case(text))
    token = stemming(remove_stopword(tokenize(text)))
    return token


def create_dictionary(messages):
    dictionary = []
    for token in messages:
        for word in token:
            if word not in dictionary:
                dictionary.append(word)
    return dictionary


def create_features(token, dictionary):
    features = np.zeros(len(dictionary))
    for word in token:
        if word in dictionary:
            features[dictionary.index(word)] += 1
    return features


def predict(text, model, dictionary):
    token = preprocess_text(text)
    features = create_features(token, dictionary).reshape(1, -1)
    prediction = model.predict(features)
    # inverse from 0 or 1 to 'ham' or 'spam'
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls


def train_model():
    DATASET_PATH = './2cls_spam_text_cls.csv'
    df = pd.read_csv(DATASET_PATH)
    messages = df['Message'].values.tolist()
    labels = df['Category'].values.tolist()

    messages = [preprocess_text(message) for message in messages]
    dictionary = create_dictionary(messages)
    X = np.array([create_features(token, dictionary) for token in messages])

    le = LabelEncoder()
    y = le.fit_transform(labels)

    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    SEED = 0

    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=VAL_SIZE,
                                                      shuffle=True,
                                                      random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                        test_size=TEST_SIZE,
                                                        shuffle=True,
                                                        random_state=SEED)

    model = GaussianNB()
    model = model.fit(X_train, y_train)
    return model, le, dictionary


# Streamlit app interface
st.title("Spam/Ham Email Classification Using Naive Bayes")

# Train model
model, le, dictionary = train_model()

# Text input
user_input = st.text_area("Enter the email content here:")

if st.button("Classify"):
    if user_input:
        result = predict(user_input, model, dictionary)
        # Display the result
        st.write(f"The email is classified as: **{result}**")
    else:
        st.write("Please enter some text to classify.")
