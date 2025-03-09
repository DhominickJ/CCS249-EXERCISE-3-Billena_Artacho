
# Created by: Billena Dhominick John and Artacho, Cristopher Ian 

# Creating the Imports
import wikipediaapi as wk
from nltk.util import bigrams as nltk_bigrams
from nltk.util import trigrams as nltk_trigrams
from nltk.tokenize import word_tokenize
from collections import Counter
from math import log2

# Getting the RE: Zero Wikipedia Page
wiki = wk.Wikipedia(user_agent='firefox', language='en')
page = wiki.page("Rem (Re:Zero)")
text = page.summary[:1000]

print(f"\nTitle: {page.title}\n")
# print(f"Summary: {page.summary[:1000]}")
print(text[:1200])

# Intial values
LAPLACE_VALUE = 1

class Bigrams:
    def __init__(self, text, laplace_value):
        self.text = text
        self.laplace_value = laplace_value

    def bigram_probabilities(self):
        tokens = word_tokenize(text.lower())
        bigram_counts = Counter(nltk_bigrams(tokens))
        unigram_counts = Counter(tokens)

        bigram_probs = {bigram: (count + self.laplace_value) / (unigram_counts[bigram[0]] * self.laplace_value + len(unigram_counts)) for bigram, count in bigram_counts.items()}

        return bigram_probs

    def bigrams_predict_next_word(self, bigram_probs, current_word):
        # Retrieve the items from the bigram model that matches the first of the bigram with the current word 
        candidates = { k[1]: v for k, v in bigram_probs.items() if k[0] == current_word }
        if not candidates:
            return None

        # get the key-value pair with the highest probability
        candidates = max(candidates, key=candidates.get)
        return candidates

    def evaluate_bigrams(self, bigram_probs, test_text):
        tokens = word_tokenize(test_text.lower())
        test_bigrams = list(nltk_bigrams(tokens))

        # Evaluate the perplexity of the model:
        log_likelihood = 0
        total_tokens = len(test_bigrams)

        for bigram in test_bigrams:
            if bigram in bigram_probs:
                probability = bigram_probs[bigram]
                log_likelihood += -1 * log2(probability)

        perplexity = 2 ** (log_likelihood / total_tokens)
        print(f"Bigram model perplexity: {perplexity:.2f} (Noice)")


class Trigrams: 
    def __init__(self, text, laplace_value):
        self.text = ""
        self.laplace_value = 1

    def trigrams_probabilities(self):
        tokens = word_tokenize(text.lower())
        trigram_counts = Counter(nltk_trigrams(tokens))
        unigram_counts = Counter(tokens)

        trigram_probs = {trigram: (count + self.laplace_value) / (unigram_counts[trigram[0]] + self.laplace_value * len(unigram_counts)) for trigram, count in trigram_counts.items()}

        return trigram_probs

    def trigrams_predict_next_word(self, trigram_probs, current_word):
        # Retrieve the items from the bigram model that matches the first of the bigram with the current word 
        candidates = { k[1]: v for k, v in trigram_probs.items() if k[0] == current_word }
        if not candidates:
            return None

        # get the key-value pair with the highest probability
        candidates = max(candidates, key=candidates.get)
        return candidates

    def evaluate_trigrams(self, trigrams_probs, test_text):
        tokens = word_tokenize(test_text.lower())
        test_trigrams = list(nltk_trigrams(tokens))

        # Evaluate the perplexity of the model:
        log_likelihood = 0
        total_tokens = len(test_trigrams)

        for trigram in test_trigrams:
            if trigram in trigrams_probs:
                probability = trigrams_probs[trigram]
                log_likelihood += -1 * log2(probability)

        perplexity = 2 ** (log_likelihood / total_tokens)
        print(f"Trigram model perplexity: {perplexity:.2f}")

# Bigrams
print("\nBIGRAMS MODEL: ")
bigrams = Bigrams(text, LAPLACE_VALUE)
wiki_bigram = bigrams.bigram_probabilities()

# Predict the next word
predicted_word = bigrams.bigrams_predict_next_word(wiki_bigram, "best")
print(f"\nPredicted next word after the word \"best\": {predicted_word}")

test_words = ["she", "won", "the", "best"]
for word in test_words:
    word.lower()
    next_word = bigrams.bigrams_predict_next_word(wiki_bigram, word)
    print(f"After '{word}', predicted word: {next_word}")

# Test the evaluation on a sample of the text
bigrams.evaluate_bigrams(wiki_bigram, text[200:700])

print("\n")
# Trigrams
print("TRIGRAMS MODEL: ")
trigrams = Trigrams(text, LAPLACE_VALUE)
wiki_trigrams = trigrams.trigrams_probabilities()

# Predicting the next words
predicting_word = trigrams.trigrams_predict_next_word(wiki_trigrams, "best")
print(f"\nPredicted next word after the word \"best\": {predicted_word}")

test_words = ["she", "won", "the", "best"]
for word in test_words:
    word.lower()
    next_word = trigrams.trigrams_predict_next_word(wiki_trigrams, word)
    print(f"After '{word}', predicted word: {next_word}")

# Test the evaluation on a sample of the text
trigrams.evaluate_trigrams(wiki_trigrams, text[200:700])
