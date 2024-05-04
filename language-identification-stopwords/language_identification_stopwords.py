from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def print_accuracy(target_validation, text_validation):
    # Merge dataframes on a common index or key column
    merged_df = pd.merge(target_validation, text_validation, on='id', suffixes=('_gt', '_pred'))

    # Compare the 'lang' column from ground truth and prediction
    correct_predictions = merged_df[merged_df['lang_gt'] == merged_df['lang_pred']]

    # Calculate accuracy
    accuracy = len(correct_predictions) / len(merged_df)
    print("Accuracy:", accuracy)

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    target_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(text_validation["text"])
    text_validation["lang"] = predictions
    text_validation = text_validation[["id", "lang"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    text_validation.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

    #print_accuracy(target_validation, text_validation)
