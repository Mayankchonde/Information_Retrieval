#installing 
!pip install beautifulsoup4 scikit-learn nltk
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
def extract_text(urls):
    data = []
    for url in urls:
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        data.append(article_text)
    return data
def process_and_vectorize(data):
    sentences = [sent_tokenize(text.lower()) for text in data]
    flat_sentences = [sentence for sublist in sentences for sentence in sublist]
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in flat_sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_sentences)
    return vectorizer, tfidf_matrix, flat_sentences

# Function to perform search
def perform_search(user_input, vectorizer, tfidf_matrix, flat_sentences):
    stop_words = set(stopwords.words('english'))
    user_input = ' '.join([word for word in user_input.split() if word not in stop_words])
    user_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
    most_relevant_sentence_index = similarity_scores.argmax()
    return flat_sentences[most_relevant_sentence_index]
st.title("Stock Market Search Engine")
user_query = st.text_input("Enter your query:")
stock_urls = ['https://www.indiainfoline.com/knowledge-center/share-market/share-market-investment-guide-for-beginners']

data = extract_text(stock_urls)

vectorizer, tfidf_matrix, flat_sentences = process_and_vectorize(data)

if user_query:
    result = perform_search(user_query, vectorizer, tfidf_matrix, flat_sentences)
    st.subheader("Most relevant information:")
    st.write(result)
