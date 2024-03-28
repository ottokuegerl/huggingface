import os
import datetime
from transformers import pipeline, AutoTokenizer

FILENAME = "hf_output_optimized.txt"
TEXTS = [
    "we love huggingface",
    "we hate microsoft",
    "What color is the sky?",
    "Welche Farbe hat der Himmel?",
]
# Specify the model and tokenizer explicitly
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def clear_screen():
    command = "cls" if os.name == "nt" else "clear"
    os.system(command)


def tokenize_texts(tokenizer, texts):
    return [tokenizer.encode(text, return_tensors="pt").tolist() for text in texts]


def analyze_sentiments(sentiment_pipeline, texts):
    return [sentiment_pipeline(text) for text in texts]


def write_results_to_file(filename, texts, tokenized_texts, results):
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "w") as file:
        for text, tokenized_text, result in zip(texts, tokenized_texts, results):
            file.write(
                f"{text} \nTokenized: {tokenized_text}\n{result}\nd o n e at: ---> {timestamp_str} <---\n"
            )


if __name__ == "__main__":
    clear_screen()
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        tokenized_texts = tokenize_texts(tokenizer, TEXTS)
        for text, tokens in zip(TEXTS, tokenized_texts):
            print(f"Tokenized input for '{text}': {tokens}")

        results = analyze_sentiments(sentiment_pipeline, TEXTS)
        for text, result in zip(TEXTS, results):
            print(f"\n---> {text} <--- \n{result} \n")

        write_results_to_file(FILENAME, TEXTS, tokenized_texts, results)

    except Exception as e:
        print(f"An error occurred: {e}")
