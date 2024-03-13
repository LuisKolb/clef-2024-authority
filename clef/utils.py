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


def write_trec_format_output(filename: str, data: List[List[Union[str, int, float]]], tag: str) -> None:
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


def combine_rumors_with_trec_file_judgements(jsons, trec_judgements_path):
    """
    create a list of RankedDocs objects in key retrieved_evidence from TREC-formatted file

    Parameters:
        - jsons: list of json-like rumors from dataset
        - trec_judgements_path: filepath to TREC-formatted file containing rank and score

    Returns:
    list of json-like rumors from dataset, with the key retrieved_evidence populated with a list of RankedDocs-like objects
    """
    trec_by_id = {}

    with open(trec_judgements_path, 'r') as file:
        for line in file:
            rumor_id, _, evidence_id, rank, score, _ = line.split('\t')
            if rumor_id not in trec_by_id:
                trec_by_id[rumor_id] = {}
            trec_by_id[rumor_id][evidence_id] = (rank, score)

    for i, item in enumerate(jsons):
        item['retrieved_evidence'] = []

        doc_ranks = trec_by_id[item['id']]
        for author_account, tweet_id, tweet_text in item['timeline']:
            if tweet_id in doc_ranks:
                rank, score = doc_ranks[tweet_id]
                item['retrieved_evidence'] += [{
                    'author_account': author_account,
                    'authority_tweet_id': tweet_id,
                    'doc_text': tweet_text,
                    'rank': int(rank),
                    'score': float(score),
                }]
        
        jsons[i] = item
    return jsons


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

