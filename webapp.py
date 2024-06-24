import streamlit as st
from transformers import pipeline
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to store the positions of capitalized words
def store_capitalization(text):
    capitalized_words = {}
    words = word_tokenize(text)
    for i, word in enumerate(words):
        if word[0].isupper():
            capitalized_words[i] = word
    return capitalized_words

# Function to restore capitalization based on stored positions
def restore_capitalization(summary, capitalized_words):
    words = word_tokenize(summary)
    for i in capitalized_words:
        if i < len(words):
            words[i] = capitalized_words[i]
    # Capitalize the first letter of each sentence
    sentences = sent_tokenize(' '.join(words))
    sentences = [sentence.capitalize() for sentence in sentences]
    return ' '.join(sentences)

# Function for minimal text cleaning
def clean_text(text):
    """
    Performs minimal text cleaning for summarization.
    - Lowercasing
    - Removing extra whitespace
    """
    text = text.lower()  # Lowercase all characters
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

# Load Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Replace with your chosen model

def summarize(article_text, summary_length):
    """
    Summarizes the article text using Hugging Face model.
    """
    capitalized_words = store_capitalization(article_text)
    cleaned_text = clean_text(article_text)  # Apply minimal cleaning

    # Set summary parameters based on selected length
    if summary_length == "Brief":
        max_length = 50
        min_length = 25
    elif summary_length == "Detailed":
        max_length = 200
        min_length = 100
    else:  # Default or Custom
        max_length = 150
        min_length = 50

    summary = summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = summary[0]["summary_text"]
    summary_text = restore_capitalization(summary_text, capitalized_words)
    return summary_text

def calculate_rouge_scores(reference, summary):
    """
    Calculates ROUGE scores between reference and generated summary.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Function to plot ROUGE scores
def plot_rouge_scores(rouge_scores):
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    precision = [rouge_scores['rouge1'].precision, rouge_scores['rouge2'].precision, rouge_scores['rougeL'].precision]
    recall = [rouge_scores['rouge1'].recall, rouge_scores['rouge2'].recall, rouge_scores['rougeL'].recall]
    f1_score = [rouge_scores['rouge1'].fmeasure, rouge_scores['rouge2'].fmeasure, rouge_scores['rougeL'].fmeasure]

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.25
    index = range(len(metrics))

    bar1 = ax.bar(index, precision, bar_width, label='Precision')
    bar2 = ax.bar([i + bar_width for i in index], recall, bar_width, label='Recall')
    bar3 = ax.bar([i + 2 * bar_width for i in index], f1_score, bar_width, label='F1-score')

    ax.set_xlabel('ROUGE Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('ROUGE Scores Comparison')
    ax.set_xticks([i + bar_width for i in index])
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on top of bars
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    st.pyplot(fig)

# Streamlit App Code

# Title and Introduction
st.title("PubMed Article Summarization")
st.write("This web application helps you summarize PubMed articles.")

# User Input/Upload Selection
user_choice = st.radio("Select input method:", ("Enter Text", "Upload File"))

# Initialize article_text
article_text = ""

# Input Text Area
if user_choice == "Enter Text":
    article_text = st.text_area("Paste your PubMed article text here:", height=200)

# Upload File
elif user_choice == "Upload File":
    uploaded_file = st.file_uploader("Upload a PubMed article (.txt)", type="txt")
    if uploaded_file is not None:
        article_text = uploaded_file.read().decode("utf-8")

# Display any errors
else:
    st.error("Please select an input method.")

# Summary Length/Style Selection
summary_length = st.selectbox("Select Summary Length/Style:", ("Brief", "Detailed", "Default"))

# Button to Summarize and Display Results
if st.button("Summarize"):
    if article_text:
        summary_text = summarize(article_text, summary_length)
        st.subheader("Original Text")
        st.write(article_text)
        st.subheader("Summary")
        st.write(summary_text)

        # Calculate ROUGE scores with a reference (original text or another summary)
        rouge_scores = calculate_rouge_scores(article_text, summary_text)

        # Display ROUGE scores in an expander
        with st.expander("View ROUGE Scores"):
            st.write("ROUGE-1: Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}".format(
                rouge_scores['rouge1'].precision, rouge_scores['rouge1'].recall, rouge_scores['rouge1'].fmeasure))
            st.write("ROUGE-2: Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}".format(
                rouge_scores['rouge2'].precision, rouge_scores['rouge2'].recall, rouge_scores['rouge2'].fmeasure))
            st.write("ROUGE-L: Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}".format(
                rouge_scores['rougeL'].precision, rouge_scores['rougeL'].recall, rouge_scores['rougeL'].fmeasure))

        # Plot ROUGE scores
        plot_rouge_scores(rouge_scores)
    else:
        st.error("Please provide text for summarization.")