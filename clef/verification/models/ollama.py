import json
import ollama

system_message = """
You are a helpful assistant doing simple reasoning tasks.
You need to decide if a statement by a given source either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES"), or if the premise is not related to the claim ("NOT ENOUGH INFO"). 
USE ONLY THE PROVIDED STATEMENT TO MAKE YOUR DECISION.
You must also provide a confidence score between 0 and 1, indicating how confident you are in your decision.
Format your answer in JSON format, like this: {"decision": ["SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO"], "confidence": [0...1]}
No yapping. Only respond in the format provided in the previous sentence!
""" 

def get_completion(input_message, model_string) -> ollama.ChatResponse:
    response = ollama.chat(
        model=model_string,
        stream=False,
        format='',
        options={
            'seed': 0,
            'temperature': 0,
        },
        messages=[
            {
                'role': 'system',
                'content': system_message,
            },
            {
                'role': 'user',
                'content': input_message,
            },
        ])

    return response # type: ignore
    
def inference_llama3(statement: str, evidence: str, model_string: str = 'instruct'):
    input_text = f'The statement: "{evidence}"\nThe claim: "{statement}"'

    valid_labels = [
        "REFUTES",
        "NOT ENOUGH INFO",
        "SUPPORTS"
    ]
    
    result = get_completion(input_text, f'llama3:{model_string}')

    try:
        answer = result['message']['content']
    except:
        try:
            # sometimes answer will rarely formatted like '{"decision": "REFUTES", "confidence": 0.9}\n\nReasoning:...'
            answer = str(result).split('\n')[0] 
        except: 
            print(f'=====\nERROR: could not unpack response string from llama3:{model_string} model:\n{result}\n======')
            return ("NOT ENOUGH INFO", 1.0)

    try:
        if answer:
            decision, confidence = json.loads(answer).values()
        else:
            print(f'====\nERROR: answer was empty, parsed from result:\n{result}\n=====')
    except ValueError:
        print(f'=====\nERROR: could not json-parse response from llama3:{model_string} model:\n{answer}\n======')
        return ("NOT ENOUGH INFO", 1.0)

    if decision in valid_labels:
        return (decision, confidence)
    else:
        return ("NOT ENOUGH INFO", 1.0)