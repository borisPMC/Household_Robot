import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

def get_custom(filename="medicine_intent.csv"):

    # Step 1: Load the data
    # Replace 'file.csv' with your file path and 'label1', 'label2' with your column names
    df = pd.read_csv(filename)

    # Step 2: Create a contingency table
    heatmap_data = pd.crosstab(df['Language'], df['Label'])

    # Step 3: Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Distribution of Class vs Language in Custom Dataset")
    plt.xlabel("Label")
    plt.ylabel("Language")
    plt.savefig(fname="Custom Dataset Statistics.png")


def get_whisper_training(logged_step=300):
    
    filename = "borisPMC/whisper_grab_medicine_intent/checkpoint-{}/trainer_state.json".format(logged_step)
    with open(filename) as f:
        data = json.load(f)
        history = data["log_history"]

    state = {
        "cer": [],
        "step": [],
        "runtime": [],
    }

    for item in history:
        if "eval_CER: " in item:
            state["cer"].append(item["eval_CER: "])
            state["step"].append(item["step"])
            state["runtime"].append(item["eval_runtime"])

    # Performance Plot
    plt.plot(state["step"][1:], state["cer"][1:])
    plt.xlabel("Steps")  # add X-axis label
    plt.ylabel("Character Error Rate")  # add Y-axis label
    plt.title("Whisper Performance by Training Steps /w CV17")  # add title
    plt.savefig(fname="graph/Whisper_performance.png")

    plt.clf()

    # Runtime Plot
    plt.plot(state["step"][1:], state["runtime"][1:])
    plt.xlabel("Steps")  # add X-axis label
    plt.ylabel("Runtime (s)")  # add Y-axis label
    plt.title("Whisper Evaluation Time by Training Steps /w CV17")  # add title
    plt.savefig(fname="graph/Whisper_runtime.png")


def main():
    get_whisper_training(180)

if __name__ == "__main__":
    main()