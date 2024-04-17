from pathlib import Path
import textstat

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
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    text_validation = text_validation.set_index("id")
    text_validation["readability_score"] = text_validation["text"].apply(lambda x: textstat.automated_readability_index(x))

    # classifying the data
    prediction = (
        text_validation["text"]
        .str.contains("delve", case=False)
        .astype(int)
    )

    prediction[text_validation["readability_score"] >= 14] = 1

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

    # For score calulation in Codespace or Dev-Container. Disable when submitting.
    #print_scores(prediction, targets_validation)