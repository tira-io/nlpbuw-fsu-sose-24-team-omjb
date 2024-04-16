from pathlib import Path
import textstat

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory


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
