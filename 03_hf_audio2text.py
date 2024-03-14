"""
################################################
## audio-to-text
## on behalf "openai/whisper-large-v3"
################################################

This script demonstrates how to use the Hugging Face transformers library
for audio transcription, specifically utilizing a pre-trained model from
the OpenAI Whisper family for Automatic Speech Recognition (ASR).
The script is structured to transcribe audio files, record the time taken
for the operation, and manage the output on the console and in a file.

automatic speech recognition (ASR) and speech translation; trained on
1 million hours of weakly labeled audio and 4 million hours of pseudolabeled
audio and 680k hours of labelled data
"""

import os
import platform
import time
from transformers import pipeline
import datetime

FILENAME = "hf_asr.txt"
AUDIO_FILES = ["99_audio_1.mp3", "99_audio_2.flac", "99_audio_3.aac", "99_audio_4.mp3"]


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def check_files_exist(file_paths):
    missing_files = [file for file in file_paths if not os.path.exists(file)]
    if missing_files:
        clear_screen()
        print(f"Missing files: ---> {', '.join(missing_files)} <---")
        return False
    return True


if __name__ == "__main__":
    if not check_files_exist(AUDIO_FILES):
        print(
            f"Please make sure --> ALL <-- audio files are available before running the script!\n"
        )
    else:
        start_time = time.time()  # Start time measurement
        clear_screen()

        print(
            f"starting automatic speech recognition (ASR) and speech translation...\n"
        )

        # the AutomaticSpeechRecognitionPipeline has a chunk_length_s parameter which is
        # helpful for working on really long audio files (for example, subtitling entire movies
        # or hour-long videos) that a model typically cannot handle on its own
        # transcriber = pipeline(model="openai/whisper-large-v2", chunk_length_s=30, return_timestamps=True)
        transcriber = pipeline(
            model="openai/whisper-large-v3",
            chunk_length_s=30,
            return_timestamps=True,
            device_map="auto",
        )
        # transcriber = pipeline(model="openai/whisper-large-v3", device_map="auto")

        # Transcribe an audio file from an url
        # transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

        # Transcribe the local audio file
        all_results = []  # Initialize a list to hold the results of each transcription

        # Process each audio file individually
        for audio_file in AUDIO_FILES:
            print(f"Processing ---> {audio_file}...")
            result = transcriber(audio_file)
            all_results.append(result)
            print(f"Completed  ---> {audio_file}\n")

        # After processing all files
        print(f"All transcriptions completed. Results: {all_results}")

        # Get the current timestamp
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"---> {timestamp_str} <---")

        # Open the file in write mode and write the result
        with open(FILENAME, "w") as file:
            file.write(f"{all_results} \nd o n e at: ---> {timestamp_str} <---\n")

        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time: ---> {execution_time:.2f} <--- seconds")
