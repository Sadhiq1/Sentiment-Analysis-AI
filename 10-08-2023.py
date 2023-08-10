#Task: Building a Simple Sentiment Lexicon using Python
labeled_data = [
    ("I am feeling happy", "positive"),
    ("This movie is disappointing", "negative"),
    ("The joyful event made me smile", "positive"),
    ("I felt sad after watching the show", "negative"),
]
def find_related_words(seed_word):
    if seed_word == "happy":
        return ["content", "cheerful", "pleased"]
    elif seed_word == "sad":
        return ["unhappy", "depressed", "miserable"]
    else:
        return []

positive_seed_words = ["happy", "joyful", "excellent"]
negative_seed_words = ["sad", "angry", "disappointing"]

from collections import defaultdict

word_freq = defaultdict(lambda: [0, 0])

for sentence, sentiment in labeled_data:
    for word in sentence.split():
        if sentiment == "positive":
            word_freq[word][0] += 1
        elif sentiment == "negative":
            word_freq[word][1] += 1

expanded_lexicon = dict()

for word in positive_seed_words:
    expanded_lexicon[word] = "positive"
    related_words = find_related_words(word)
    for related_word in related_words:
        expanded_lexicon[related_word] = "positive"

for word in negative_seed_words:
    expanded_lexicon[word] = "negative"
    related_words = find_related_words(word)
    for related_word in related_words:
        expanded_lexicon[related_word] = "negative"
print("Expanded Sentiment Lexicon:")
for word, sentiment in expanded_lexicon.items():
    print(f"{word}: {sentiment}")
