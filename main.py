from multiprocessing import Value
import re
import threading
import time
from queue import Queue
import numpy as np
from transformers import pipeline, Pipeline

def listen_audio(asr_pipe: Pipeline, nlp_pipe: Pipeline, label_queue: Queue, finding_medicine) -> None:

    import sounddevice as sd
    while True:
        # Avoid listening to audio if finding_medicine is True
        if finding_medicine.value:
            print("Finding medicine, please wait (Audio)...")
            time.sleep(11)  # Simulate the 10-second duration for finding a medicine
            continue

        audio_array = None
        transcript = None
        class_label = None

        print("Recording for 5 seconds...\n")
        audio_data = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished
        audio_array = np.squeeze(audio_data)  # Convert to 1D array

        transcript = asr_pipe(audio_array)
        class_label = nlp_pipe(transcript)

        print("Transcript:", transcript["text"])

        # Put the label in the queue
        if bool(re.search(r"[^a-zA-Z0-9\s'\u4e00-\u9fff]", transcript["text"])):
            label_queue.put("Empty") # Put "Empty" in the queue if the transcript contains special characters, avoid accident inputs.
        else:
            label_queue.put(class_label["label"])
        
        # Wait 1 second before looping again
        time.sleep(1)

def find_user_thread(user_flag, finding_medicine) -> None:
    while True:

        # Avoid listening to audio if finding_medicine is True
        if finding_medicine.value:
            print("Finding medicine, please wait (User)")
            time.sleep(11)  # Simulate the 10-second duration for finding a medicine
            continue

        # Simulate finding a user
        time.sleep(5)  # Simulate the 5-second duration for finding a user
        user_flag.value = find_user()

        # Wait 1 second before looping again
        time.sleep(1)

def find_user():
    # Simulate finding a user
    return True

def find_medicine(class_label, finding_medicine, label_queue) -> None:
    print("Finding medicine:", class_label, "\n")
    time.sleep(10)  # Simulate the 10-second duration for finding a medicine
    print("Medicine found!\n")

    # Reset flag
    finding_medicine.value = False
    label_queue.put("Empty")  # Put "Empty" in the queue to reset the label
    return

def main():
    # Initialize ASR and NLP pipelines
    asr_pipe = pipeline("automatic-speech-recognition", model="borisPMC/whisper_small_grab_medicine_intent")
    nlp_pipe = pipeline("text-classification", model="borisPMC/bert_grab_medicine_intent", tokenizer="bert-base-multilingual-uncased")

    # Shared variables and queues
    user_flag = Value('b', True)  # Shared boolean flag for user detection
    finding_medicine = Value('b', False)  # Shared boolean flag for finding medicine
    label_queue = Queue()  # Queue to store labels from the audio process

    # Create threads
    user_thread = threading.Thread(target=find_user_thread, args=(user_flag, finding_medicine), daemon=True)
    audio_thread = threading.Thread(target=listen_audio, args=(asr_pipe, nlp_pipe, label_queue, finding_medicine), daemon=True)

    # Start threads
    user_thread.start()
    audio_thread.start()
    class_label = "Empty"

    try:
        while True:
            class_label = label_queue.get()  # Get the label from the queue
            if user_flag.value and class_label != "Empty":
                finding_medicine.value = True
                find_medicine(class_label, finding_medicine=finding_medicine, label_queue=label_queue)  # Trigger find_medicine() if conditions are met
            else:
                finding_medicine.value = False
                print("No executable commands detected, please request again.\n")
    
    except KeyboardInterrupt:
        print("Stopping threads...")
        


if __name__ == "__main__":
    main()