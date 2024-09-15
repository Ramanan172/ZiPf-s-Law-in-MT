import nltk
import string
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re


def read_text_from_file(file_path):
    # Reads text from a file.
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# File paths
sinhala_file = 'Resources/sinhala_sentences.txt'
english_file = 'Resources/english_sentences.txt'

# Read sentences from files
sinhala_sentence = read_text_from_file(sinhala_file)
english_sentence = read_text_from_file(english_file)


def preprocess_text(text):
    # Preprocesses text by tokenizing, lower casing, and removing punctuation.
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    return tokens


# Preprocess the text
sinhala_tokens = preprocess_text(sinhala_sentence)
english_tokens = preprocess_text(english_sentence)

# Print the tokens
print("Sinhala Tokens: ", sinhala_tokens)
print("English Tokens: ", english_tokens)


def calculate_word_frequencies(tokens):
    # Calculates the frequency of each word in a list of tokens.
    word_frequencies = Counter(tokens)
    return dict(word_frequencies)


# Example usage
sinhala_frequencies = calculate_word_frequencies(sinhala_tokens)
english_frequencies = calculate_word_frequencies(english_tokens)

print("Frequency of Sinhala word: ", sinhala_frequencies)
print("Frequency of English word: ", english_frequencies)


def plot_word_frequencies(frequencies, language):
    # Plots a bar chart of word frequencies for a given language.

    # 1. Create a DataFrame from the frequency dictionary
    df = pd.DataFrame.from_dict(frequencies, orient='index', columns=['Frequency'])

    # 2. Sort the DataFrame by Frequency in descending order
    df = df.sort_values(by='Frequency', ascending=False)

    # 3. Create a bar chart
    df.plot(kind='bar', title=f'{language} Word Frequencies')

    # 4. Show the plot
    plt.show()


# Plot the word frequencies for Sinhala and English
plot_word_frequencies(sinhala_frequencies, 'Sinhala')
plot_word_frequencies(english_frequencies, 'English')


def analyze_zipfs_law(frequencies):
    # Analyzes Zipf's Law for the top 25 words in a corpus.
    sorted_words = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    top_25 = sorted_words[:25]
    ranks = list(range(1, len(top_25) + 1))
    f_r_products = [f * r for r, (word, f) in zip(ranks, top_25)]
    df = pd.DataFrame({
        "Rank": ranks,
        "Word": [word for word, freq in top_25],
        "Frequency": [freq for word, freq in top_25],
        "f*r": f_r_products
    })
    return df


sinhala_zipf_analysis = analyze_zipfs_law(sinhala_frequencies)
english_zipf_analysis = analyze_zipfs_law(english_frequencies)

print("Zipf's Analysis for Sinhala")
print(sinhala_zipf_analysis)
print("Zipf's Analysis for English")
print(english_zipf_analysis)
