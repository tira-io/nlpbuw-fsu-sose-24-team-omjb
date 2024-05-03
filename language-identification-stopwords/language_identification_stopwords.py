from collections import Counter
import os
from pathlib import Path
import re
from nltk.tokenize import word_tokenize

from tqdm import tqdm
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

lang_ids = [
    "af",
    "az",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "hr",
    "it",
    "ko",
    "nl",
    "no",
    "pl",
    "ru",
    "ur",
    "zh",
]

def get_lang_common_words_df_list(rows: int):
    # Assuming your list of text files is stored in a directory named 'text_files'
    directory = 'commonwords'
    file_list = os.listdir(directory)

    dataframes = {}

    for filename in file_list:
        if filename.endswith('.txt'):  # Assuming all files are text files
            file_path = os.path.join(directory, filename)
            # Read text file into a dataframe
            df = pd.read_csv(file_path, skiprows=1, names=['word'], nrows=rows)  # Adjust sep parameter if needed
            # Add dataframe to the dictionary with filename as key
            dataframes[filename.split('-')[1].split('.')[0]] = df
    return dataframes

def calculate_percentage(text, top_words):
    tokenized_text = tokenize(text)
    text_word_count = len(tokenized_text)
    common_words_count = sum(tokenized_text.count(word) for word in top_words)
    return (common_words_count / text_word_count) * 100 if text_word_count != 0 else 0

# Tokenize function
def tokenize(text):
    return word_tokenize(text.lower())

def find_max(df):
    # Function to find max value and corresponding column name
    def find_max(row):
        row_data = row.drop('id')  # Exclude 'id' column
        max_value = row_data.max()
        max_column = row_data.idxmax()
        return max_value, max_column

    # Apply the function to each row
    df[['max_value', 'lang']] = df.apply(find_max, axis=1, result_type='expand')
    return df[['id', 'lang']]

def calculate_word_appearance_percentage(text_df, language_df_map):
    # Initialize empty DataFrame
    result_data = []

    # Iterate through texts
    for index, row in text_df.iterrows():
        # Tokenize text
        #tokenized_text = tokenize(text)
        #text_word_count = len(tokenized_text)
        
        # Calculate word frequencies
        #word_freq = Counter(tokenized_text)

        new_entry = {}
        new_entry['id'] = row['id']
        
        for lang in lang_ids:
            language_dataset = language_df_map[lang]
            new_entry[lang] = calculate_percentage(row['text'], language_dataset['word'])

        # Append results to DataFrame
        result_data.append(new_entry)

    # Save or use result_data DataFrame as needed
    return pd.DataFrame(result_data)

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    df_list = get_lang_common_words_df_list(25)
    result = calculate_word_appearance_percentage(text_validation, df_list)
    prediction = find_max(result)

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
