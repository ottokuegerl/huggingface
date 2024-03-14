"""
This script demonstrates a simple use case of the Hugging Face transformers
library for sentiment analysis, along with some basic file operations and
datetime usage. 
"""

import os
import platform
from transformers import pipeline
import datetime


FILENAME = "hf_fileinfo.txt"
TXT = "we love huggingface"
TXT2 = "we hate microsoft"


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


if __name__ == "__main__":
    clear_screen()
    # Initialize the sentiment-analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    print(f"\nstart sentiment-analysis...")

    # Perform sentiment analysis
    result = sentiment_pipeline(TXT)
    print(f"\n---> {TXT} <--- \n")
    print(f"{result} \n")

    result2 = sentiment_pipeline(TXT2)
    print(f"---> {TXT2} <--- \n")
    print(f"{result2} \n")

    # Convert the result to a string format
    # Note: The result is a list of dictionaries, so we convert it to a string to write to a file
    result_str = str(result)
    result2_str = str(result2)

    # Get the current timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"---> {timestamp_str} <---")

    # Open the file in write mode and write the result
    with open(FILENAME, "w") as file:
        file.write(f"{TXT} \n")
        file.write(f"{result_str}\nd o n e at: ---> {timestamp_str} <---\n")
        file.write(f"{TXT2} \n")
        file.write(f"{result2_str}\nd o n e at: ---> {timestamp_str} <---\n")
