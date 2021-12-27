import datetime as dt
import re
import numexpr as ne
import numpy as np
import pandas as pd
import streamlit as st
import s3fs
import os

from flair.data import Sentence
from flair.models import TextClassifier
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings, BertEmbeddings

def main():
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Sentiment Analysis', 'Reddit Mental Health NLP Search'])

    if page == 'Homepage':
        st.title("Welcome!")
        st.write(
            """
This is a basic site, built using [Streamlit](https://www.streamlit.io/) entirely in Python, to demo some of the Machine Learning related projects created by [Brian Newton, PsyD](https://www.linkedin.com/in/briannewtonpsyd/).

The models were trained using a GPU and deployed to an AWS EC2 instance for CPU inference. Models are currently being cached for quick inference, but future iterations will leverage Sagemaker and API endpoints.

Please choose a section from the dropdown in the left navbar, each one represents a different ML project. This area will also contain contextual information about the project.

            """
        )
    elif page == 'Sentiment Analysis':
        st.sidebar.title("Data")
        st.sidebar.info(
        "The data for this project was obtained from a dataset which contained sentences from an english language learning website. "
        "Each sentence was labeled with one of 6 emotions (Anger, Surprise, Sadness, Happiness, Disgust, Fear) or No Emotion. "
        "No Emotion labeled sentences were later removed as it improved the detection sensitivity of the model."
        )
        st.sidebar.title("Model")
        st.sidebar.info(
        "The ML model for this project uses [flairNLP](https://github.com/flairNLP/flair) to generate a text classification model using distilbert-base-uncased word embeddings alongside the news-forward and news-backward embeddings provided by flairNLP. "
        "It then trains a neural network uses those embeddings to identify the associated labels in the data above. The current model achieved an F1 micro score .877 after 5 epochs."
        )

        mhsearch_text_input = ''
        # Set page title
        st.title('Sentence Sentiment Analysis')

        # Load classification model
        with st.spinner('Loading classification model...'):
            classifier = load_model('best-model.pt')

        ### SINGLE TWEET CLASSIFICATION ###
        st.subheader('This project uses Natural Language processing to predict the emotion expressed by a sentence.')
        st.write('Type a sentence here and the model will attempt to determine the emotion expressed.')

        # Get sentence input, preprocess it, and convert to flair.data.Sentence format
        sentiment_text_input = st.text_input('A sentence to classify:')

        if sentiment_text_input != '':
            # Pre-process tweet
            sentence = Sentence(preprocess(sentiment_text_input))

            # Make predictions
            with st.spinner('Hold on, determining the emotion...'):
                classifier.predict(sentence)

            # Show predictions
            label_dict = {'0': 'No emotion', '1': 'Anger', '2': 'Disgust', '3': 'Fear', '4': 'Happiness', '5': 'Sadness', '6': 'Surprise'}

            if len(sentence.labels) > 0:
                st.write('I think it\'s:')
                st.write(label_dict[sentence.labels[0].value] + ' with ', int(sentence.labels[0].score*100), '% confidence')
    else:
        st.sidebar.title("Data")
        st.sidebar.info(
        "The data for this project was obtained by scraping (using a Python package which accesses the reddit API) the top 1000 reddit posts from each of the 10 most active mental health related subreddits. "
        "The dataset includes the subreddit name, the post title, the post text, the post link and the text of the \"best\" comment."
        )
        st.sidebar.title("Model")
        st.sidebar.info(
        "The ML model for this project uses [flairNLP](https://github.com/flairNLP/flair) on a GPU to generate word embeddings for the post title for the 10000 posts in the dataset using bert-based-uncased from HuggingFace's Transformers along with news-forward and news-backward embeddings from flairNLP."
        "The embeddings are then stored alongside each reddit post and stored in a compressed HDF5 file."
        "This site then generates embeddings for the sentence entered and performs a consine similarity calculation between the generated embeddings and every post title, returning the top 10 posts."
        )
        sentiment_text_input = ''
        st.title('Search Reddit MH Posts')

        with st.spinner('Loading sentence embeddings...'):
            embeddings = load_embeddings()
            embeddingsdf = load_embeddings_df()
            
        ### SINGLE TWEET CLASSIFICATION ###
        st.subheader('The project uses Natural Language Processing to find reddit post titles from mental health subreddits that are most similar to the entered sentence.')
        st.write('Type a brief description of a mental health related scenario and the model will return similar reddit posts.')

        # Get sentence input, preprocess it, and convert to flair.data.Sentence format
        mhsearch_text_input = st.text_input('A sentence to search:')

        if mhsearch_text_input != '':
            # Pre-process tweet
            sentence = Sentence(preprocess(mhsearch_text_input))

            # Make predictions
            with st.spinner('Hold on, finding similar posts...'):
                cosines = read_and_compare_embeddings(embeddings, mhsearch_text_input, embeddingsdf['embeddings'])
                embeddingsdf['cosine'] = cosines
                finaldf = embeddingsdf.sort_values(by=['cosine'], ascending=False)
                resultsdf = finaldf[0:10]

            if len(resultsdf) > 0:
                st.write('Here are some posts that might be similar:')
                for index, row in resultsdf.iterrows():
                    st.markdown('From r/' + row['Subreddit'] + ': [' + row['PostTitle'] + '](' + row['PostURL'] + ') (' + str(int(row['cosine'][0][0] * 100)) + "%)")

@st.cache(allow_output_mutation=True)
def load_embeddings_df():
    return pd.read_hdf('models/depression_embeddings.h5', key="embeddingsdf")

@st.cache(allow_output_mutation=True)
def load_embeddings():
    return DocumentPoolEmbeddings([BertEmbeddings('bert-base-uncased'), FlairEmbeddings('news-backward'), FlairEmbeddings('news-forward')])

def read_and_compare_embeddings(embeddings, query, embeddingscolumn):
    # your query
    query = Sentence(query)

    # embed query
    embeddings.embed(query)

    embedding = query.embedding.cpu().data.numpy().reshape(1, -1)
    
    cosinesarray = []
    for row in embeddingscolumn:
        cosinesarray.append(cosine_vectorized(row.reshape(1, -1), embedding))
    
    return cosinesarray

def cosine_vectorized(array1, array2):
    sumyy = np.einsum('ij,ij->i',array2,array2)
    sumxx = np.einsum('ij,ij->i',array1,array1)[:,None]
    sumxy = array1.dot(array2.T)
    sqrt_sumxx = ne.evaluate('sqrt(sumxx)')
    sqrt_sumyy = ne.evaluate('sqrt(sumyy)')
    return ne.evaluate('(sumxy/sqrt_sumxx)/sqrt_sumyy')

@st.cache(allow_output_mutation=True)
def load_model(model):
    # Create connection object.
    # `anon=False` means not anonymous, i.e. it uses access keys to pull data.
    fs = s3fs.S3FileSystem(anon=False)
    
    with fs.open(model) as f:
        classifier = TextClassifier.load(f)
        return classifier

def preprocess(text):
    # Preprocess function
    allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
    punct = '!?,.@#'
    maxlen = 280

    # Delete URLs, cut to maxlen, space out punction with spaces, and remove unallowed chars
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])

if __name__ == '__main__':
    main()
