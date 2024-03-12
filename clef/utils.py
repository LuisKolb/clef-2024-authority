from typing import List, Tuple, Dict, TypedDict, Union, Optional, NamedTuple
import textwrap
import json 
import os

class RumorDict(TypedDict):
    id: str
    rumor: str
    label: str
    timeline: List[List[str]]
    evidence: List[List[str]]
    retrieved_evidence: List[List[Union[str, int, float]]]

class RankedDocs(NamedTuple):
    author_account: str
    authority_tweet_id:str
    doc_text: str
    rank: int
    score: float


def load_rumors_from_jsonl(filepath: Union[str, os.PathLike]) -> List[RumorDict]:
    jsons = []
    with open(filepath, encoding='utf8') as file:
        for line in file:
            jsons += [json.loads(line)]
    return jsons


def write_trec_format_output(filename: str, data: Dict[str, List[List[str]]], tag: str) -> None:
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
        for rumor_id in data.keys():
            for author_account, authority_tweet_id, doc_text, rank, score in data[rumor_id]:
                line = f"{rumor_id}\tQ0\t{authority_tweet_id}\t{rank}\t{score}\t{tag}\n"
                file.write(line)


def write_jsonlines_from_dicts(filename: Union[str, os.PathLike], dicts: List[Dict]) -> None:
    with open(filename, 'w') as file:
        for item in dicts:
            file.write(f'{json.dumps(item)}\n')






def wrap(text: str):
    """
    Print strings line-wrapped.

    Parameters:
    - text (str): The text you want to print.
    """

    wrapped_text = textwrap.fill(text, width=80) #text is the object which you want to print
    print(wrapped_text)

