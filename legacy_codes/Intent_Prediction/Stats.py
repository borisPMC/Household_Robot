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
    plt.figure(figsize=(16, 12))
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

def readJson(fpath):

    # Opening JSON file
    f = open(fpath)

    # returns JSON object as a dictionary
    data = json.load(f)["log_history"]

    train_loss =  [0]
    intent_loss = []
    ner_loss = []
    intent_f1 = []
    med_f1_seq = []
    tkn_f1 = []

    # Iterating through the json list
    for i in data:
        keys = i.keys()
        if "loss" in keys:
            train_loss.append(i["loss"])
        if "eval_intent_f1" in keys:
            intent_f1.append(i["eval_intent_f1"])
        if "eval_intent_loss" in keys:
            intent_loss.append(i["eval_intent_loss"])
        if "eval_ner_loss" in keys:
            ner_loss.append(i["eval_ner_loss"])
        if "eval_ner_med_acc" in keys:
            med_f1_seq.append(i["eval_ner_med_acc"])
        if "eval_ner_tkn_f1" in keys:
            tkn_f1.append(i["eval_ner_tkn_f1"])

    pd.DataFrame({
        "Train Loss": train_loss,
        "Eval Intent Loss": intent_loss,
        "Eval NER Loss": ner_loss,
        "Intent F1": intent_f1,
        "Med List SeqF1": med_f1_seq,
        "Token F1": tkn_f1,
    }).to_csv("./temp/result.csv")

    # Closing file
    f.close()

def main():
    get_custom()

if __name__ == "__main__":
    readJson("./models/multitask_model/checkpoint-1740/trainer_state.json")