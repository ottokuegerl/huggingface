"""
This script demonstrates a simple use case of the Hugging Face transformers
library for sentiment analysis, along with some basic file operations and
datetime usage. 
"""

import os
import platform
from transformers import (
    pipeline,
    AutoTokenizer,
)  # transformers --> Hugging Face Transformers Library
import datetime

# Specify the model and tokenizer explicitly
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

FILENAME = "hf_output.txt"
TXT = "we love huggingface"
TXT2 = "we hate microsoft"
TXT3 = "Welche Farbe hat der Himmel?"


def clear_screen():
    try:
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")
    except Exception as e:
        print(f"Error clearing screen: {e}\n")


if __name__ == "__main__":
    clear_screen()

    print(f"\nstart sentiment-analysis...")

    try:
        # Initialize the sentiment-analysis pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)

        # Initialize the tokenizer with the model used in the sentiment pipeline
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"---> Error <---\n")
        print(f"Failed to initialize sentiment analysis pipeline or tokenizer: {e}\n")
        print(f"---> Error <---\n")
        exit()

    try:
        # Tokenize the input text
        input_ids = tokenizer.encode(TXT, return_tensors="pt")
        input_ids2 = tokenizer.encode(TXT2, return_tensors="pt")
        input_ids3 = tokenizer.encode(TXT3, return_tensors="pt")

        print(f"Tokenized input for '{TXT}': {input_ids.tolist()}")
        print(f"Tokenized input for '{TXT2}': {input_ids2.tolist()}")
        print(f"Tokenized input for '{TXT3}': {input_ids3.tolist()}")
    except Exception as e:
        print(f"---> Error <---\n")
        print(f"error during tokenization: {e}\n")
        print(f"---> Error <---\n")

    try:
        # Perform sentiment analysis
        result = sentiment_pipeline(TXT)
        print(f"\n---> {TXT} <--- \n")
        print(f"{result} \n")

        result2 = sentiment_pipeline(TXT2)
        print(f"---> {TXT2} <--- \n")
        print(f"{result2} \n")

        result3 = sentiment_pipeline(TXT3)
        print(f"---> {TXT3} <--- \n")
        print(f"{result3} \n")
    except Exception as e:
        print(f"---> Error <---\n")
        print(f"error during sentiment analysis: {e}\n")
        print(f"---> Error <---")

    # Convert the result to a string format and write it to an file
    # Note: The result is a list of dictionaries, so we convert it to a string to write to a file
    result_str = str(result)
    result2_str = str(result2)
    result3_str = str(result3)

    # Get the current timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"---> {timestamp_str} <---")

    try:
        # Open the file in write mode and write the result
        with open(FILENAME, "w") as file:
            file.write(f"{TXT} \n")
            file.write(f"Tokenized: {input_ids.tolist()}\n")
            file.write(f"{result_str}\nd o n e at: ---> {timestamp_str} <---\n")
            file.write(f"{TXT2} \n")
            file.write(f"Tokenized: {input_ids2.tolist()}\n")
            file.write(f"{result2_str}\nd o n e at: ---> {timestamp_str} <---\n")
            file.write(f"{TXT3} \n")
            file.write(f"Tokenized: {input_ids3.tolist()}\n")
            file.write(f"{result3_str}\nd o n e at: ---> {timestamp_str} <---\n")
    except Exception as e:
        print(f"---> Error <---\n")
        print(f"Failed to write to file: {FILENAME}: {e}\n")
        print(f"---> Error <---")
