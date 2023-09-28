import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords from NLTK
import nltk
nltk.download('stopwords')

# Function to extract text from URLs
def extract_text(urls):
    data = []
    for url in urls:
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        data.append(article_text)
    return data

# Function to process and vectorize text
def process_and_vectorize(data):
    # Tokenize sentences
    sentences = [sent_tokenize(text.lower()) for text in data]

    # Flatten the list of sentences
    flat_sentences = [sentence for sublist in sentences for sentence in sublist]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in flat_sentences]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_sentences)
    
    return vectorizer, tfidf_matrix, flat_sentences

# Function to perform search
def perform_search(user_input, vectorizer, tfidf_matrix, flat_sentences):
    # Download stopwords from NLTK
    stop_words = set(stopwords.words('english'))
    
    # Vectorize user input
    user_input = ' '.join([word for word in user_input.split() if word not in stop_words])
    user_tfidf = vectorizer.transform([user_input])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)

    # Get the most relevant sentence
    most_relevant_sentence_index = similarity_scores.argmax()

    # Return the most relevant sentence
    return flat_sentences[most_relevant_sentence_index]

# Streamlit App
st.title("Stock Market Search Engine")

# Input for user query
user_query = st.text_input("Enter your query:")

# URLs for stock market information (you can add more URLs)
stock_urls = ['https://www.indiainfoline.com/knowledge-center/share-market/share-market-investment-guide-for-beginners']

# Extract text from URLs
data = extract_text(stock_urls)

# Process and vectorize text
vectorizer, tfidf_matrix, flat_sentences = process_and_vectorize(data)

# Perform search when the user enters a query
if user_query:
    result = perform_search(user_query, vectorizer, tfidf_matrix, flat_sentences)
    st.subheader("Most relevant information:")
    st.write(result)
