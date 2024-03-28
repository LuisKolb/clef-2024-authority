from transformers import pipeline, Pipeline
from typing import NamedTuple

class VerificationResult(NamedTuple):
    label: str
    score: float

# Initialize the NLI pipeline with a pre-trained model
roberta_pipeline: Pipeline = pipeline("text-classification", model="roberta-large-mnli")

def inference_roberta(claim: str, evidence: str) -> VerificationResult:
    input_text = f"{evidence} [SEP] {claim}"

    # Use the NLI pipeline to predict the relationship
    result = roberta_pipeline(input_text)

    label_map = {
        "CONTRADICTION": "REFUTES",
        "NEUTRAL": "NOT ENOUGH INFO",
        "ENTAILMENT": "SUPPORTS"
    }

    label = label_map[result[0]['label']]
    score = result[0]['score']

    # Return the result
    return (label, score)