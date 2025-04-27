import os
import random
import nltk

# Download NLTK tokenizer if needed
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from textblob import TextBlob
#from docx import Document  # For processing Word documents



# Define homophone errors
HOMOPHONE_ERRORS = {
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

# Define punctuation errors
PUNCTUATION_ERRORS = [",", ".", "!", "?", ";", ":", "-", "_", "(", ")"]

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
    return HOMOPHONE_ERRORS.get(word.lower(), word)

# Function to introduce punctuation errors
def introduce_punctuation(sentence):
    words = list(sentence)
    if len(words) > 5:
        index = random.randint(1, len(words) - 2)
        words[index] = random.choice(PUNCTUATION_ERRORS)
    return "".join(words)

# Function to introduce grammatical errors
def introduce_grammar_errors(sentence):
    blob = TextBlob(sentence)
    words = blob.words
    if len(words) > 2:
        index = random.randint(0, len(words) - 2)
        words[index] = introduce_typo(words[index])
    return " ".join(words)


def introduce_real_grammar_errors(sentence):
    words = word_tokenize(sentence)
    if len(words) < 3:
        return sentence  # Not enough words to manipulate

    error_type = random.choice(["subject_verb", "tense", "preposition", "article", "plurality"])

    if error_type == "subject_verb":
        # Force wrong subject-verb agreement
        for i, word in enumerate(words):
            if word.lower() in ["is", "are", "was", "were", "has", "have"]:
                words[i] = random.choice(["is", "are", "was", "were", "has", "have"])
                break

    elif error_type == "tense":
        # Swap past tense verbs to present or vice versa
        for i, word in enumerate(words):
            if word.endswith("ed"):
                words[i] = word.rstrip("ed")  # Remove 'ed' to simulate wrong tense
                break

    elif error_type == "preposition":
        # Replace common prepositions incorrectly
        prepositions = {"in": "on", "on": "at", "at": "in", "to": "for", "for": "to", "with": "by"}
        for i, word in enumerate(words):
            if word.lower() in prepositions:
                words[i] = prepositions[word.lower()]
                break

    elif error_type == "article":
        # Remove articles ("a", "an", "the")
        words = [w for w in words if w.lower() not in ["a", "an", "the"]]

    elif error_type == "plurality":
        # Make plural nouns singular or vice versa (simple way)
        for i, word in enumerate(words):
            if word.endswith("s"):
                words[i] = word.rstrip("s")  # Remove plural
                break
            elif len(word) > 3 and word.isalpha():
                words[i] = word + "s"  # Wrongly pluralize
                break

    return " ".join(words)


# Function to introduce formatting errors
def introduce_formatting(sentence):
    return sentence.replace(" ", "") if random.random() > 0.5 else sentence.replace(" ", "  ")

# Function to apply noise at different levels
def apply_noise(text, noise_level="moderate"):
    words = word_tokenize(text)
    new_words = []

    for word in words:
        if random.random() < (0.05 if noise_level == "light" else 0.10 if noise_level == "moderate" else 0.20):
            error_type = random.choices(
                ["typo", "homophone", "formatting"],
                weights=[0.6,0.2,0.2])[0]
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

# Define paths (adjust according to your VS Code workspace)
queries_file = "queries.txt"  # Text file containing queries
input_doc_folder = "documents/"  # Folder with original .docx files
output_doc_folder = "noisy_documents/"  # Folder for noisy documents

# Choose noise levels
query_noise_level = "light"
doc_noise_level = "moderate"

# Load and process queries
queries = load_queries(queries_file)
noisy_queries = process_queries(queries, query_noise_level, "noisy_queries.txt")

# Process Word documents
process_documents(input_doc_folder, output_doc_folder, doc_noise_level)

# Display some examples
print("\nSample Queries:")
for i in range(3):
    print(f"Original: {queries[i]}")
    print(f"Noisy: {noisy_queries[i]}")
    print("-" * 50)
