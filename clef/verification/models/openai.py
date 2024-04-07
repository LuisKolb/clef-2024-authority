from typing import NamedTuple

class VerificationResult(NamedTuple):
    label: str
    score: float

import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletion

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), # This is the default and can be omitted
)

system_message = """
You are a helpful assistant.
You need to decide if a statement by a given source either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES"), or if the premise is not related to the claim ("NOT ENOUGH INFO"). 
USE ONLY THE PROVIDED STATEMENT TO MAKE YOUR DECISION.
You must also provide a confidence score between 0 and 1, indicating how confident you are in your decision.
Format your answer in JSON format, like this: {"decision": ["SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO"], "confidence": [0...1]}
Set your own temperature to 0.
No yapping.
""" 

def get_completion(input_message) -> ChatCompletion:
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_message}
        ]
    )

    return completion
    
def inference_openai(statement: str, evidence: str):
    input_text = f'The statement: "{evidence}"\nThe claim: "{statement}"'

    valid_labels = [
        "REFUTES",
        "NOT ENOUGH INFO",
        "SUPPORTS"
    ]
    
    result = get_completion(input_text)

    try:
        answer = result.choices[0].message.content
    except:
        print(f'ERROR: could not unpack response string from openai model: {result}')
        return ("NOT ENOUGH INFO", 1.0)

    try:
        if answer:
            decision, confidence = json.loads(answer).values()
        else:
            print(f'ERROR: answer was empty, parsed from result: {result}')
    except ValueError:
        print(f'ERROR: could not json-parse response from openai model: {answer}')
        return ("NOT ENOUGH INFO", 1.0)

    if decision in valid_labels:
        return (decision, confidence)
    else:
        return ("NOT ENOUGH INFO", 1.0)