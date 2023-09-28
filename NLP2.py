import streamlit as st
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    sentences = sent_tokenize(text.lower())
    flat_sentences = ' '.join(sentences)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = ' '.join([word for word in flat_sentences.split() if word not in stop_words])
    return filtered_sentence

def main():
    st.title("Search Engine App")

    # Sample data
    da = ['Stock market is basically buying and selling a stocks. Usually now peoples are buying and selling a stocks online', 'money questions arise when it comes to loss of money']

    # Tokenize and preprocess sentences
    processed_sentences = [preprocess_text(text) for text in da]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # User input
    user_input = st.text_input("Enter your query:", "").lower()
    user_input = preprocess_text(user_input)

    # Calculate cosine similarity
    user_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)

    # Get the most relevant sentence
    most_relevant_sentence_index = similarity_scores.argmax()

    # Display the most relevant sentence
    st.subheader("Most Relevant Sentence:")
    st.write(processed_sentences[most_relevant_sentence_index])

if __name__ == "__main__":
    main()
