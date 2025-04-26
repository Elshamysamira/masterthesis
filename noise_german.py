import os
import random
import nltk
from nltk.tokenize import word_tokenize
from textblob_de import TextBlobDE  # TextBlob for German
import unidecode

# Download NLTK tokenizer if needed
nltk.download('punkt')

# Define German homophone errors (common ones)
HOMOPHONE_ERRORS_GERMAN = {
    "seid": "seit",
    "seit": "seid",
    "das": "dass",
    "dass": "das",
    "wieder": "wider",
    "wider": "wieder",
    "mehr": "meer",
    "man": "mann",
    "den": "denn",
}

# Define German punctuation errors
PUNCTUATION_ERRORS = [",", ".", "!", "?", ";", ":", "-", "_", "(", ")", "..."]

# Function to introduce a typo
def introduce_typo(word):
    if len(word) > 3:
        typo_type = random.choice(["swap", "delete", "insert"])
        index = random.randint(0, len(word) - 2)

        if typo_type == "swap":
            word = word[:index] + word[index + 1] + word[index] + word[index + 2:]
        elif typo_type == "delete":
            word = word[:index] + word[index + 1:]
        elif typo_type == "insert":
            random_char = random.choice("abcdefghijklmnopqrstuvwxyzäöüß")
            word = word[:index] + random_char + word[index:]
    return word

# Function to introduce a homophone error
def introduce_homophone(word):
    return HOMOPHONE_ERRORS_GERMAN.get(word.lower(), word)

# Function to introduce punctuation errors
def introduce_punctuation(sentence):
    words = list(sentence)
    if len(words) > 5:
        index = random.randint(1, len(words) - 2)
        words[index] = random.choice(PUNCTUATION_ERRORS)
    return "".join(words)

# Function to introduce grammatical errors
def introduce_grammar_errors(sentence):
    blob = TextBlobDE(sentence)
    words = blob.words
    if len(words) > 2:
        index = random.randint(0, len(words) - 2)
        words[index] = introduce_typo(words[index])
    return " ".join(words)

# Function to introduce formatting errors
def introduce_formatting(sentence):
    return sentence.replace(" ", "") if random.random() > 0.5 else sentence.replace(" ", "  ")

# Function to apply noise at different levels
def apply_noise(text, noise_level="moderate"):
    words = word_tokenize(text, language="german")
    new_words = []

    for word in words:
        if random.random() < (0.05 if noise_level == "light" else 0.10 if noise_level == "moderate" else 0.20):
            error_type = random.choice(["typo", "homophone", "formatting"])
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

    return noisy_text

# Function to read queries from a file
def load_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        queries = [line.strip() for line in file.readlines()]
    return queries

# Function to process and save noisy queries
def process_queries(queries, noise_level, output_file):
    noisy_queries = [apply_noise(query, noise_level) for query in queries]
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(noisy_queries))
    return noisy_queries

# Function to process TXT documents
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

# Paths
queries_file = "queries.txt"
input_doc_folder = "documents/"
output_doc_folder = "noisy_documents/"

# Noise levels
query_noise_level = "light"
doc_noise_level = "moderate"

# Execution
queries = load_queries(queries_file)
noisy_queries = process_queries(queries, query_noise_level, "noisy_queries.txt")

process_documents(input_doc_folder, output_doc_folder, doc_noise_level)

# Display examples
print("\nSample Queries:")
for i in range(min(3, len(queries))):
    print(f"Original: {queries[i]}")
    print(f"Noisy: {noisy_queries[i]}")
    print("-" * 50)
