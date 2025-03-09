
# Created by: Billena Dhominick John and Artacho, Cristopher Ian 

# Creating the Imports
import wikipediaapi as wk
from nltk import bigrams
from nltk.util import trigrams
from nltk.tokenize import word_tokenize
from collections import Counter

# Getting the RE: Zero Wikipedia Page
wiki = wk.Wikipedia(user_agent='firefox', language='en')
page = wiki.page("Rem (Re:Zero)")
text = page.summary

print(f"Title: {page.title}")
# print(f"Summary: {page.summary[:1000]}")
print(text[:1200])


class Bigrams:
    def __init__(self, text, laplace_value, test_text)
        self.text = ""
        self.laplace_value = 1
        self.test_text = []

    def bigram_probabilities(self.text, self.laplace_value = 1):
        tokens = word_tokenize(text.lower())
        bigram_counts = Counter(bigrams(tokens))
        unigram_counts = Counter(tokens)

        bigram_probs = {bigram: (count + laplace_value) / (unigram_counts[bigram[0]] + len(unigram_counts)) for bigram, count in bigram_counts.items()}

        return bigram_probs

    def bigrams_predict_next_word(bigram_probs, current_word):
        # Retrieve the items from the bigram model that matches the first of the bigram with the current word 
        candidates = { k[1]: v for k, v in bigram_probs.items() if k[0] == current_word }
        if not candidates:
            return None

        # get the key-value pair with the highest probability
        candidates = max(candidates, key=candidates.get)
        return candidates

    def evaluate_bigrams(bigram_probs, self.test_text):
        tokens = word_tokenize(test_text.lower())
        test_bigrams = list(bigrams(tokens))
        correct = 0
        total = 0
        
        for i in range(len(test_bigrams)):
            current_word = test_bigrams[i][0]
            actual_next = test_bigrams[i][1]
            predicted = bigrams_predict_next_word(bigram_probs, current_word)
            
            if predicted == actual_next:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Bigram model accuracy: {accuracy:.2%}")

class Trigrams: 
    def __init__(self, text, laplace_value, test_text)
        self.text = ""
        self.laplace_value = 1
        self.trigrams_probs = []
        self.test_text = []

    def trigram_probabilities(self.text, self.laplace_value = 1):
        tokens = word_tokenize(text.lower())
        trigram_counts = Counter(trigrams(tokens))
        unigram_counts = Counter(tokens)

        trigram_probs = {trigram: (count + laplace_value) / (unigram_counts[trigrams[0]] + len(unigram_counts)) for trigrams, count in trigram_counts.items()}

        return trigram_probs

    def trigram_predict_next_word(trigam_probs, current_word):
        # Retrieve the items from the bigram model that matches the first of the bigram with the current word 
        candidates = { k[1]: v for k, v in trigram_probs.items() if k[0] == current_word }
        if not candidates:
            return None

        # get the key-value pair with the highest probability
        candidates = max(candidates, key=candidates.get)
        return candidates

    def evaluate_trigram(bigram_probs, self.test_text):
        tokens = word_tokenize(test_text.lower())
        test_trigrams = list(trigrams(tokens))
        correct = 0
        total = 0
        
        for i in range(len(test_trigrams)):
            current_word = test_trigrams[i][0]
            actual_next = test_trigrams[i][1]
            predicted = trigrams_predict_next_word(trigrams_probs, current_word)
            
            if predicted == actual_next:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Trigrams model accuracy: {accuracy:.2%}")

bigrams = Bigrams(text, laplace_value)
wiki_bigram = bigrams.bigram_probabilities(text)
print(wiki_bigram)

# Predict the next word
predicted_word = bigrams.bigrams_predict_next_word(wiki_bigram, "best")
print(f"\nPredicted next word after the word \"best\": {predicted_word}")

# Test bigram model
test_words = ["she", "won", "the", "best"]
for word in test_words:
    word.lower()
    next_word = bigrams.bigrams_predict_next_word(wiki_bigram, word)
    print(f"After '{word}', predicted word: {next_word}")

# Evaluate bigram model

# Test the evaluation on a sample of the text
evaluate_bigrams(wiki_bigram, text[300:600])