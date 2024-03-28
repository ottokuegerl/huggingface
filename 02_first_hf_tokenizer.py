import os
import datetime
from transformers import pipeline, AutoTokenizer

# Constants
FILENAME = "hf_tokenizer_output.txt"
TEXTS = [
    "Welche Farbe hat der Himmel?",
    "Wie spät ist es?",
    "Welcher Tag ist heute?",
    "Erstelle ein Bild von einer lustigen Ente mit Brille und einem Apfel?",
    "HPE ist super",
    "NVIDIA ist auch super",
    "Microsoft ist keine tolle Firma",
]
# Specify the model and tokenizer explicitly
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Initialization
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def clear_screen():
    command = "cls" if os.name == "nt" else "clear"
    os.system(command)


def process_texts(texts):
    # Tokenization and sentiment analysis
    tokenized_texts = [
        tokenizer.encode(text, return_tensors="pt").tolist() for text in texts
    ]
    sentiment_results = sentiment_pipeline(texts)

    # Writing results to file with timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FILENAME, "w") as file:
        for text, tokenized_text, result in zip(
            texts, tokenized_texts, sentiment_results
        ):
            token_count = len(tokenized_text[0])  # Calculate the number of tokens
            """
            this calculation takes into account the full list of tokens,
            including special tokens added by the tokenizer (e.g., [CLS], [SEP] for BERT-based models
            """
            output_str = (
                f"{text}\n"
                f"Tokenized: {tokenized_text} \n{token_count} tokens will be used as input\n"
                f"{result}\nDone at: ---> {timestamp_str} <---\n"
            )
            print(output_str)  # Print to console
            file.write(output_str)  # Write to file


if __name__ == "__main__":
    clear_screen()
    try:
        process_texts(TEXTS)
        print(f"Processing complete. Results written to ---> {FILENAME} <---\n")
    except Exception as e:
        print(f"An error occurred: {e}")
