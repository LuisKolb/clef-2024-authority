from transformers import pipeline, Pipeline
from typing import NamedTuple

class VerificationResult(NamedTuple):
    label: str
    score: float

# Initialize the NLI pipeline with a pre-trained model
bart_pipeline: Pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def inference_bart(claim: str, evidence: str) -> VerificationResult:
    # Define the candidate labels for NLI
    candidate_labels = ["contradiction","neutral","entailment"]

    # Use the NLI pipeline to predict the relationship
    results = bart_pipeline(evidence, hypothesis=claim, candidate_labels=candidate_labels, multi_label=False)
    
    label_map = {
        "contradiction": "REFUTES",
        "neutral": "NOT ENOUGH INFO",
        "entailment": "SUPPORTS"
    }

    label = label_map[results['labels'][0].lower()]
    score = results['scores'][0]

    return (label, score)