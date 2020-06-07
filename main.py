import datetime as dt
import re

import pandas as pd
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier

def main():
    
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Sentiment Analysis', 'N/A'])

    if page == 'Homepage':
        st.title('The Homepage')
        st.subheader('Nothing to see here yet.')
    elif page == 'Sentiment Analysis':
        # Set page title
        st.title('Sentence Sentiment Analysis')

        # Load classification model
        with st.spinner('Loading classification model...'):
            classifier = load_model('models/best-model.pt')

        # Preprocess function
        allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
        punct = '!?,.@#'
        maxlen = 280

        def preprocess(text):
            # Delete URLs, cut to maxlen, space out punction with spaces, and remove unallowed chars
            return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])

        ### SINGLE TWEET CLASSIFICATION ###
        st.subheader('Type a sentence here and the model will attempt to determine the emotion expressed.')

        # Get sentence input, preprocess it, and convert to flair.data.Sentence format
        text_input = st.text_input('Text:')

        if text_input != '':
            # Pre-process tweet
            sentence = Sentence(preprocess(text_input))

            # Make predictions
            with st.spinner('Hold on, determining the emotion...'):
                classifier.predict(sentence)

            # Show predictions
            label_dict = {'0': 'No emotion', '1': 'Anger', '2': 'Disgust', '3': 'Fear', '4': 'Happiness', '5': 'Sadness', '6': 'Surprise'}

            if len(sentence.labels) > 0:
                st.write('I think it\'s:')
                st.write(label_dict[sentence.labels[0].value] + ' with ', int(sentence.labels[0].score*100), '% confidence')
    else:
        st.title('Another page')
        st.subheader('Nothing to see here yet.')


def load_model(model):
    classifier = TextClassifier.load(model)
    
    return classifier

if __name__ == '__main__':
    main()