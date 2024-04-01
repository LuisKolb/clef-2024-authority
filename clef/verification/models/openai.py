from typing import NamedTuple

class VerificationResult(NamedTuple):
    label: str
    score: float

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), # This is the default and can be omitted
)

system_message = 'You are a helpful assistant. You need to decide if a statement by a given source either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES") or if the premise is not related to the claim ("NOT ENOUGH INFO"). No yapping.'

def get_completion(input_message):
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_message}
        ]
    )

    return completion.choices[0].message
    
def inference_openai(statement: str, evidence: str) -> VerificationResult:
    input_text = f'The statement: "{evidence}"\nThe claim: "{statement}"'

    result = get_completion(input_text).content

    valid_labels = [
        "REFUTES",
        "NOT ENOUGH INFO",
        "SUPPORTS"
    ]

    if result in valid_labels:
        return (result, 1.0)
    else:
        return ("NOT ENOUGH INFO", 1.0)