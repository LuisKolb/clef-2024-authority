from typing import List, Tuple, Dict

def write_trec_format_output(filename: str, data: List[Tuple[str, int, int, float]], tag: str) -> None:
    """
    Writes data to a file in the TREC format.

    Parameters:
    - filename (str): The name of the file to write to.
    - data (List[Tuple[str, int, int, float]]): A list of tuples, where each tuple contains:
        - rumor_id (str): The unique ID for the given rumor.
        - authority_tweet_id (int): The unique ID for the authority tweet.
        - rank (int): The rank of the authority tweet ID for that given rumor_id.
        - score (float): The score given by the model for the authority tweet ID.
    - tag (str): The string identifier of the team/model.
    """
    with open(filename, 'w') as file:
        for rumor_id, authority_tweet_id, rank, score in data:
            line = f"{rumor_id}\tQ0\t{authority_tweet_id}\t{rank}\t{score}\t{tag}\n"
            file.write(line)



def write_jsonlines_from_dicts(filename: str, dicts: List[Dict]) -> None:
    import json

    with open(filename, 'w') as file:
        for item in dicts:
            print(type(item))
            
            file.write(f'{json.dumps(item)}\n')

import textwrap

def wrap(text: str):
    """
    Print strings line-wrapped.

    Parameters:
    - text (str): The text you want to print.
    """
    wrapped_text = textwrap.fill(text, width=80) #text is the object which you want to print
    print(wrapped_text)