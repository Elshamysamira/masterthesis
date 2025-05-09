import os
import random
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob  # TextBlob for English
import unidecode

nltk.download('punkt')

# Define English homophone errors
HOMOPHONE_ERRORS_ENGLISH = {
    "their": "there",
    "there": "their",
    "effect": "affect",
    "accept": "except",
    "principal": "principle",
    "weather": "whether",
    "your": "you're",
    "its": "it's",
    "lose": "loose",
}

# Define English punctuation errors
PUNCTUATION_ERRORS = [",", ".", "!", "?", ";", ":", "-", "_", "(", ")", "..."]

# Function to introduce a typo
def introduce_typo(word):
    if len(word) > 2:
        typo_type = random.choice(["swap", "delete", "insert"])
        index = random.randint(0, len(word) - 2)

        if typo_type == "swap":
            word = word[:index] + word[index + 1] + word[index] + word[index + 2:]
        elif typo_type == "delete":
            word = word[:index] + word[index + 1:]
        elif typo_type == "insert":
            random_char = random.choice("abcdefghijklmnopqrstuvwxyz")
            word = word[:index] + random_char + word[index:]
    return word

# Function to introduce a homophone error
def introduce_homophone(word):
    return HOMOPHONE_ERRORS_ENGLISH.get(word.lower(), word)

# Function to introduce punctuation errors
def introduce_punctuation(sentence):
    words = list(sentence)
    if len(words) > 5:
        index = random.randint(1, len(words) - 2)
        words[index] = random.choice(PUNCTUATION_ERRORS)
    return "".join(words)

# Function to introduce simple grammar typos
def introduce_grammar_errors(sentence):
    blob = TextBlob(sentence)
    words = blob.words
    if len(words) > 2:
        index = random.randint(0, len(words) - 2)
        words[index] = introduce_typo(words[index])
    return " ".join(words)

# NEW: Function to introduce real grammatical errors in English
def introduce_real_grammar_errors(sentence):
    words = word_tokenize(sentence, language="english")
    if len(words) < 3:
        return sentence

    error_type = random.choice(["subject_verb", "tense", "article", "plural", "preposition"])

    if error_type == "subject_verb":
        # Force wrong subject-verb agreement
        for i, word in enumerate(words):
            if word.lower() in ["is", "are", "was", "were", "has", "have"]:
                words[i] = random.choice(["is", "are", "was", "were", "has", "have"])
                break

    elif error_type == "tense":
        # Change past tense to present (simple version)
        for i, word in enumerate(words):
            if word.endswith("ed"):
                words[i] = word.rstrip("ed")
                break

    elif error_type == "article":
        # Remove or wrong article
        articles = {"a": "the", "an": "a", "the": "an"}
        for i, word in enumerate(words):
            if word.lower() in articles:
                words[i] = articles[word.lower()]
                break

    elif error_type == "plural":
        # Make plural wrong
        for i, word in enumerate(words):
            if word.endswith("s"):
                words[i] = word.rstrip("s")
                break
            elif len(word) > 3 and word.isalpha():
                words[i] = word + "s"
                break

    elif error_type == "preposition":
        prepositions = {"in": "on", "on": "at", "at": "in", "to": "for", "for": "to", "with": "by"}
        for i, word in enumerate(words):
            if word.lower() in prepositions:
                words[i] = prepositions[word.lower()]
                break

    return " ".join(words)

# Function to introduce formatting errors
def introduce_formatting(sentence):
    return sentence.replace(" ", "") if random.random() > 0.5 else sentence.replace(" ", "  ")

# Function to apply noise
def apply_noise(text, noise_level="moderate"):
    words = word_tokenize(text, language="english")
    new_words = []

    for word in words:
        if random.random() < (0.05 if noise_level == "light" else 0.10 if noise_level == "moderate" else 0.20):
            error_type = random.choices(
                ["typo", "homophone", "formatting"],
                weights=[0.6, 0.2, 0.2])[0]
            if error_type == "typo":
                new_words.append(introduce_typo(word))
            elif error_type == "homophone":
                new_words.append(introduce_homophone(word))
            elif error_type == "formatting":
                new_words.append(introduce_formatting(word))
        else:
            new_words.append(word)

    noisy_text = " ".join(new_words)

    if random.random() < (0.02 if noise_level == "light" else 0.05 if noise_level == "moderate" else 0.10):
        noisy_text = introduce_punctuation(noisy_text)

    if random.random() < (0.02 if noise_level == "light" else 0.05 if noise_level == "moderate" else 0.10):
        noisy_text = introduce_grammar_errors(noisy_text)

    if random.random() < (0.02 if noise_level == "light" else 0.05 if noise_level == "moderate" else 0.10):
        noisy_text = introduce_real_grammar_errors(noisy_text)

    return noisy_text

# Function to read queries
def load_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        queries = [line.strip() for line in file.readlines()]
    return queries

# Function to process and save noisy queries
def process_queries(queries, noise_level, output_file):
    # Define how many errors per noise level
    noise_to_error_count = {
        "light": 1,
        "moderate": 2,
        "heavy": 3,
    }
    
    errors_per_query = noise_to_error_count.get(noise_level, 2)  # Default to moderate
    
    noisy_queries = []
    
    for query in queries:
        words = word_tokenize(query, language="english")
        
        # Apply multiple errors
        for _ in range(errors_per_query):
            error_type = random.choices(["typo", "homophone", "formatting", "punctuation", "grammar", "real_grammar"],
                                        weights=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1])[0]
            
            if error_type == "typo" and words:
                index = random.randint(0, len(words) - 1)
                words[index] = introduce_typo(words[index])
            
            elif error_type == "homophone" and words:
                index = random.randint(0, len(words) - 1)
                words[index] = introduce_homophone(words[index])
            
            elif error_type == "formatting":
                query = introduce_formatting(" ".join(words))
                words = word_tokenize(query, language="english")
            
            elif error_type == "punctuation":
                query = introduce_punctuation(" ".join(words))
                words = word_tokenize(query, language="english")
            
            elif error_type == "grammar":
                query = introduce_grammar_errors(" ".join(words))
                words = word_tokenize(query, language="english")
            
            elif error_type == "real_grammar":
                query = introduce_real_grammar_errors(" ".join(words))
                words = word_tokenize(query, language="english")
        
        noisy_query = " ".join(words)
        noisy_queries.append(noisy_query)
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(noisy_queries))
    
    return noisy_queries

# Function to process documents
def process_documents(input_folder, output_folder, noise_level):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r", encoding="utf-8") as file:
                text = file.read()

            noisy_text = apply_noise(text, noise_level)

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(noisy_text)

            print(f"Processed {filename} with {noise_level} noise.")

# --- Main Execution ---

# Paths
queries_file = "queries/queries_english/questions_english.txt"
input_doc_folder = "documents/txt_english"
output_doc_folder = "documents/noisy_documents_severe_english"
output_noisy_queries_folder = "queries/noisy_queries_severe_english"

# Noise levels
query_noise_level = "heavy"
doc_noise_level = "heavy"


# Execution
queries = load_queries(queries_file)
noisy_queries_file = os.path.join("queries/noisy_queries_severe_english", "noisy_queries_severe_english.txt")
noisy_queries = process_queries(queries, query_noise_level, noisy_queries_file)

process_documents(input_doc_folder, output_doc_folder, doc_noise_level)

# Display some examples
print("\nSample Queries:")
for i in range(min(3, len(queries))):
    print(f"Original: {queries[i]}")
    print(f"Noisy: {noisy_queries[i]}")
    print("-" * 50)
