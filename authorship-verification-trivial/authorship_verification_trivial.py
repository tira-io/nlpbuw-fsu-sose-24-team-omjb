from pathlib import Path
import numpy as np
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def print_scores(expected, actual, value=False):
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

    if value:
        return accuracy
    else:
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

def get_rank(word, frequencies):
    if word in frequencies:
        return sorted(frequencies, key=lambda x: frequencies[x], reverse=True).index(word) + 1
    else:
        return len(frequencies) + 1

def add_ranking(text_train):
    # Iterate over each row
    for word in ["the", "to", "of", "and", "be"]:
        ranks = []
        for i, row in text_train.iterrows():
            frequencies = row['term_frequency']
            rank = get_rank(word, frequencies)
            ranks.append(rank)
        text_train[word + '_rank'] = ranks
    return text_train

def preprocess_df(df):
    df['id'] = pd.to_numeric(df['id'])
    df.set_index('id', inplace=True)
    df.sort_index(inplace=True)
    return df

def compute_ranks(df):
    df['term_frequency'] = df['text'].apply(calculate_term_frequency)
    df = add_ranking(df)
    df.drop(columns=['term_frequency', 'text'], inplace=True)
    return df

def random_forest_calibration(text_train, targets_train, text_validation, targets_validation):
    max_depth = np.arange(2, 8)
    n_estimators = np.arange(1, 300)
    
    acc = np.zeros((len(max_depth)+2,len(n_estimators)+1))
    
    for m in max_depth:
        for n in n_estimators:
            rf_classifier = RandomForestClassifier(n_estimators = n, max_depth=m)
            
            rf_classifier.fit(text_train, targets_train['generated'])
            
            prediction = pd.DataFrame({'id': text_validation.index})
            prediction['generated'] = rf_classifier.predict(text_validation)

            prediction = preprocess_df(prediction)

            acc[m, n] = print_scores(prediction, targets_validation, value=True)
    
            print(f"maxDepth = {m}, nTrees = {n}: Acc = {acc[m,n]}".format(m, n, acc))
    
    print(f"Höchste Accurency: {str(np.max(acc))} bei {str(np.unravel_index(np.argmax(acc),np.shape(acc)))}".format(acc))
    
    return acc

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text_train = preprocess_df(text_train)

    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = preprocess_df(targets_train)

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    text_validation = preprocess_df(text_validation)

    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = preprocess_df(targets_validation)

    # Compute Ranking of 5 most used words
    text_train = compute_ranks(text_train)
    text_validation = compute_ranks(text_validation)

    # Initializing the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=141, max_depth=7)
    rf_classifier.fit(text_train, targets_train['generated'])

    prediction = pd.DataFrame({'id': text_validation.index})
    prediction['generated'] = rf_classifier.predict(text_validation)

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
    prediction = preprocess_df(prediction)

    print_scores(prediction, targets_validation)
