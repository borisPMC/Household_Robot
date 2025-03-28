from multiprocessing import Value
import time
from queue import Queue
import numpy as np
from transformers import pipeline, Pipeline

def find_user():
    # Simulate finding a user
    return True

# Main function for the Master program
# Expected to be run forever
def find_user_thread(user_flag, stop_receiving_commands) -> None:
    while True:

        # Avoid listening to audio if stop_receiving_commands is True
        if stop_receiving_commands.value:
            print("Finding medicine, please wait (User)")
            time.sleep(11)  # Simulate the 10-second duration for finding a medicine
            continue

        # Simulate finding a user
        time.sleep(5)  # Simulate the 5-second duration for finding a user
        user_flag.value = find_user()

        # Wait 1 second before looping again
        time.sleep(1)