from pathlib import Path
import re
import pandas as pd

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def print_scores(expected, actual):
    import pandas as pd
    expected = expected.rename(columns={'generated': 'prediction'})
    merged_df = pd.merge(actual, expected, on='id')

    TP = merged_df[(merged_df['prediction'] == 1) & (merged_df['generated'] == 1)].shape[0]
    FP = merged_df[(merged_df['prediction'] == 1) & (merged_df['generated'] == 0)].shape[0]
    FN = merged_df[(merged_df['prediction'] == 0) & (merged_df['generated'] == 1)].shape[0]
    TN = merged_df[(merged_df['prediction'] == 0) & (merged_df['generated'] == 0)].shape[0]

    if (TP + FP) == 0:
        precision = 1
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("Accuracy: {:.2f}".format(accuracy))
    print("F1 Score: {:.2f}".format(f1_score))

def calculate_term_frequency(text):
    # Entferne Satzzeichen und Sonderzeichen, und wandle in Kleinbuchstaben um
    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()
    # Zerlege den Text in einzelne Wörter (Terme)
    terms = cleaned_text.split()
    # Zähle die Häufigkeit jedes einzelnen Terms
    term_frequency = {}
    for term in terms:
        term_frequency[term] = term_frequency.get(term, 0) + 1
    return term_frequency

def create_term_frequency_df(row):
    # Calculate term frequency for the current row
    term_frequency = calculate_term_frequency(row['text'])
    # Create DataFrame from term frequency dictionary
    term_frequency_df = pd.DataFrame(list(term_frequency.items()), columns=['word', 'frequency'])
    # Sort DataFrame by frequency in descending order
    term_frequency_df = term_frequency_df.sort_values(by='frequency', ascending=False).reset_index()
    # Add a column for word rank
    term_frequency_df['rank'] = range(len(term_frequency_df))
    term_frequency_df.set_index("rank")
    term_frequency_df = term_frequency_df.drop(columns=['index', 'rank'])
    return term_frequency_df

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    text_validation = text_validation.set_index("id")

    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = targets_validation.set_index("id")

    # classifying the data
    prediction = (
        text_validation["text"]
        .str.contains("delve", case=False)
        .astype(int)
    )

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

    text_validation['term_frequency'] = text_validation['text'].apply(calculate_term_frequency)

    # Create a list to store term frequency DataFrames for each row
    term_frequency_dfs = []

    print(targets_validation)
    print(targets_validation.loc[1])

    # Iterate over each row in the DataFrame
    for index, row in text_validation.iterrows():
        # Create term frequency DataFrame for the current row
        term_frequency_df = create_term_frequency_df(row)
        # Add the term frequency DataFrame to the list
        term_frequency_dfs.append(term_frequency_df)

    # Display the list of term frequency DataFrames
    for i, term_frequency_df in enumerate(term_frequency_dfs):
        print(f"Term Frequency DataFrame for Row {i}:")
        print("Generated:", targets_validation.loc[i, 'generated'])
        print(term_frequency_df)

    print_scores(prediction, targets_validation)
