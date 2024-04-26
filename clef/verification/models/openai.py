from abc import ABC, abstractmethod
import json
from typing import List, NamedTuple
from openai import OpenAI
from openai.types.chat import ChatCompletion

import re
import os
from clef.utils.data_loading import AuthorityPost

import logging
logger_verification = logging.getLogger('clef.retr')

class VerificationResult(NamedTuple):
    label: str
    score: float

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
    
def inference_openai(statement: str, evidence: str) -> VerificationResult:
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
        return VerificationResult("NOT ENOUGH INFO", 1.0)

    try:
        if answer:
            decision, confidence = json.loads(answer).values()
        else:
            print(f'ERROR: answer was empty, parsed from result: {result}')
    except ValueError:
        print(f'ERROR: could not json-parse response from openai model: {answer}')
        return VerificationResult("NOT ENOUGH INFO", 1.0)

    if decision in valid_labels:
        return VerificationResult(decision, confidence)
    else:
        return VerificationResult("NOT ENOUGH INFO", 1.0)
    

class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, claim: str, evidence: str, **kwargs) -> VerificationResult:
        """Verify a claim based on the evidence."""
        pass
    
    def __call__(self, claim: str, evidence: str, **kwargs) -> VerificationResult:
        return self.verify(claim, evidence, **kwargs)

class OpenaiVerifier(BaseVerifier):
    client: OpenAI
    model: str = "gpt-4-turbo-preview"
    valid_labels: List =  [
            "REFUTES",
            "NOT ENOUGH INFO",
            "SUPPORTS"
        ]
    system_message: str = """You are a helpful assistant doing simple reasoning tasks.
You will be given a statement and a claim.
You need to decide if a statement either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES"), or if the statement is not related to the claim ("NOT ENOUGH INFO"). 
USE ONLY THE STATEMENT AND THE CLAIM PROVIDED BY THE USER TO MAKE YOUR DECISION.
You must also provide a confidence score between 0 and 1, indicating how confident you are in your decision.
You must format your answer in JSON format, like this: {"decision": ["SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO"], "confidence": [0...1]}
No yapping.
""" 

    def __init__(self, api_key:str='') -> None:
        self.client = OpenAI(
            api_key=(api_key or os.environ.get("OPENAI_API_KEY")),
        )
    
    def get_completion(self, input_message) -> ChatCompletion:
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": input_message}
            ],
            temperature=0.2
        )

        return completion
    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        input_text = f'Statement: "{evidence}"\nClaim: "{claim}"'
      
        result = self.get_completion(input_text)

        try:
            answer = result.choices[0].message.content
        except:
            logger_verification.warn(f'could not unpack response string from openai model: {result}')
            return VerificationResult("NOT ENOUGH INFO", 1.0)

        try:
            if answer:
                decision, confidence = json.loads(answer).values()
            else:
                logger_verification.warn(f'answer was empty, parsed from result: {result}')
        except ValueError:
            logger_verification.warn(f'ERROR: could not json-parse response from openai model: {answer}')
            return VerificationResult("NOT ENOUGH INFO", 1.0)

        if decision in self.valid_labels:
            return VerificationResult(decision, confidence)
        else:
            return VerificationResult("NOT ENOUGH INFO", 1.0)

oaiver = OpenaiVerifier()