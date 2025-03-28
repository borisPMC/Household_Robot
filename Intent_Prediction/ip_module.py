import re
import time
from queue import Queue
import numpy as np
from transformers import Pipeline

# Main function for the Master program
# Expected to be run forever
def listen_audio_thread(asr_pipe: Pipeline, nlp_pipe: Pipeline, label_queue: Queue, stop_receiving_commands) -> None:

    import sounddevice as sd
    while True:
        # Avoid listening to audio if stop_receiving_commands is True
        if stop_receiving_commands.value:
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