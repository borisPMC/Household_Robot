from typing import List
import re
import pandas as pd

def convert_sentence_to_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the CSV file to update the NER_Tag column based on speech tokens and MED_TOKEN_LIST.

    Args:
        fpath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Updated dataframe with modified NER_Tag.
    """
    # Optimized table for better performance and scalability. Added item when creating the table.
    MED_TOKEN_LIST = {
        "高血壓": "122",
        "血壓藥": "122",
        "壓血壓": "122",
        "血管緊張素轉換酶抑制藥": "12222222222",
        "血管緊張素轉換酶抑制劑": "12222222222",
        "high blood pressure": "122",
        "hypertension": "1",
        "ace inhibitor": "12",
        "糖尿病": "344",
        "甲福明": "344",
        "diabetes": "3",
        "metformin": "3",
        "冠脈病": "566",
        "冠心病": "566",
        "心臟病": "566",
        "膽固醇藥": "5666",
        "血脂": "56",
        "Blood lipids": "56",
        "Blood fats": "56",
        "heart disease": "56",
        "coronary heart disease": "566",
        "cad": "5",
        "阿伐他汀": "5666",
        "atorvastatin": "5",
        "抑鬱": "78",
        "depression": "7",
        "阿米替林": "7888",
        "amitriptyline": "7",
        "antidepressant": "7",
    }

    # Process each row in the dataframe
    for index, row in df.iterrows():

        speech = row["Speech"].lower()

        # Split the speech into tokens
        tokens = hybrid_split(speech)

        # Ensure NER_Tag matches the length of tokens
        ner_tag = ["0"] * len(tokens)

        # Update NER_Tag based on MED_TOKEN_LIST matches
        for key, value in MED_TOKEN_LIST.items():
            key_tokens = hybrid_split(key)  # Split the key into tokens
            key_length = len(key_tokens)

            # Search for matching sequences of tokens
            for i in range(len(tokens) - key_length + 1):
                if tokens[i:(i + key_length)] == key_tokens:
                    # Update the corresponding NER_Tag tokens
                    ner_tag[i:(i + key_length)] = list(value[:key_length])

        # Join the NER_Tag list back into a string
        ner_tag_str = "'" + "".join(ner_tag)

        # Update the dataframe
        df.at[index, "NER_Tag"] = ner_tag_str

    return df

def detect_language(df: pd.DataFrame):

    """
    Detect the language of string by chracter. If a Chinese character exist, consider the sentence as Cantonese (based on mixed-code property of HK Cantonese.)
    """

    # Process each row in the dataframe
    for index, row in df.iterrows():

        lang = "Unknown"
        speech = row["Speech"]
        if re.search(r"[\u4e00-\ufaff]", speech, re.UNICODE):
            lang = "Cantonese"
        else:
            lang = "English"
        df.at[index, "Language"] = lang

    return df
    
def hybrid_split(string: str) -> List[str]:
    """
    Split a string into tokens using a hybrid regex.
    """
    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, string, re.UNICODE)
    return matches


# Example usage
fpath = "Intent_Prediction/multitask_audio/multitask_ds.xlsx"
out_fpath = "Intent_Prediction/multitask_audio/multitask_ds_modified.xlsx"
out_csv = "Intent_Prediction/multitask_audio/multitask_ds_modified.csv"
df = pd.read_excel(fpath)
lang_df = detect_language(df)
tokenized_df = convert_sentence_to_tokens(df)



# updated_df = convert_sentence_to_tokens(fpath)
# tokenized_df.to_excel(out_fpath, index=False)
tokenized_df.to_csv(out_csv, index=False)