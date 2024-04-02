from typing import List, Tuple, Dict, TypedDict, Union, Optional, NamedTuple
import json
import os
import re

from clef.utils.preprocessing import clean_text_basic

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
    i = 0
    if data:
        with open(filename, 'w') as file:
            for rumor_id, authority_tweet_id, rank, score in data:
                line = f"{rumor_id}\tQ0\t{authority_tweet_id}\t{rank}\t{score}\t{tag}\n"
                file.write(line)
                i += 1
        print(f'wrote {i} lines to {filename}')
    else:
        print('data was empty, nothing was written to disk')


def combine_rumors_with_trec_file_judgements(jsons, trec_judgements_path, sep='\t'):
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
            # handle edge case where dataset may contain field which contain an additional whitespace, ...
            # which was then saved together with the field value as a string looking like 'id 0 "id " 1'
            # (only seems to affect TERRIER trec files)
            line = re.sub('"', '', line)
            line = re.sub('  ', ' ', line)
            rumor_id, _, evidence_id, rank, score, _ = line.split(sep)
            if rumor_id not in trec_by_id:
                trec_by_id[rumor_id] = {}
            trec_by_id[rumor_id][evidence_id] = (rank, score)

    for i, item in enumerate(jsons):
        item['retrieved_evidence'] = []

        doc_ranks = trec_by_id[item['id']]
        for author_account, tweet_id, tweet_text in item['timeline']:
            if tweet_id in doc_ranks:
                rank, score = doc_ranks[tweet_id]
                item['retrieved_evidence'] += [[
                    author_account,
                    tweet_id,
                    tweet_text,
                    int(rank),
                    float(score),
                ]]
        
        jsons[i] = item
    return jsons


def write_jsonlines_from_dicts(filename: Union[str, os.PathLike], dicts: List[Dict]) -> None:
    with open(filename, 'w') as file:
        for item in dicts:
            file.write(f'{json.dumps(item)}\n')


def clean_jsons(jsons):
    from clef.utils.preprocessing import clean_text_basic

    data_cleaned = []

    for entry in jsons:
        
        tl_clean = []
        for account_url, tl_tweet_id, tl_tweet in entry['timeline']:
            tl_tweet_cleaned = clean_text_basic(tl_tweet)
            if tl_tweet_cleaned:
                tl_clean += [[account_url, tl_tweet_id, tl_tweet_cleaned]]

        ev_clean = []
        for account_url, ev_tweet_id, ev_tweet in entry['evidence']:
            ev_tweet_cleaned = clean_text_basic(ev_tweet)
            if ev_tweet_cleaned:
                ev_clean += [[account_url, ev_tweet_id, ev_tweet_cleaned]]

        data_cleaned += [{
            'id': entry['id'],
            'rumor': clean_text_basic(entry['rumor']),
            'label': entry['label'],
            'timeline': tl_clean,
            'evidence': ev_clean,
        }]

    return data_cleaned


def add_author_info(dataset, preprocess, add_author_bio, add_author_name, author_info_file):
    with open(author_info_file, 'r') as file:
        author_info = json.load(file)
    for item in dataset:
        timeline = item['timeline']
        new_timeline = []
        for account, tweet_id, tweet_text in timeline:
            name = author_info[account.strip()]["translated_name"]
            bio = author_info[account.strip()]["translated_bio"]

            if preprocess:
                name = clean_text_basic(name)
                bio = clean_text_basic(bio)
            
            new_tweet_text = f'Text: {tweet_text}'
            if add_author_bio:
                new_tweet_text = f'Account Description: {bio}\n' + new_tweet_text
            if add_author_name:
                new_tweet_text = f'Account: {name}\n' + new_tweet_text

            new_timeline += [[account, tweet_id, new_tweet_text]]
        item['timeline'] = new_timeline

        evidence = item['evidence']
        new_evidence = []
        for account, tweet_id, tweet_text in evidence:
            name = author_info[account.strip()]["translated_name"]
            bio = author_info[account.strip()]["translated_bio"]

            if preprocess:
                name = clean_text_basic(name)
                bio = clean_text_basic(bio)
            
            new_tweet_text = f'Text: {tweet_text}'
            if add_author_bio:
                new_tweet_text = f'Account Description: {bio}\n' + new_tweet_text
            if add_author_name:
                new_tweet_text = f'Account: {name}\n' + new_tweet_text

            new_evidence += [[account, tweet_id, new_tweet_text]]
        item['evidence'] = new_evidence
    
    return dataset


task5_dir = 'clef2024-checkthat-lab/task5' # relative to repo root
author_info_file_train = 'clef/data/train-author-data-translated.json' # relative to repo root
author_info_file_dev = 'clef/data/author-data-translated.json' # relative to repo root

def load_datasets(preprocess: bool, root_path: str, add_author_name: bool = False, add_author_bio: bool = False):
    
    data_path = os.path.join(root_path, task5_dir, 'data')

    filepath_train = os.path.join(data_path, 'English_train.json')
    filepath_dev = os.path.join(data_path, 'English_dev.json')

    train_jsons = load_rumors_from_jsonl(filepath_train)
    dev_jsons = load_rumors_from_jsonl(filepath_dev)

    print(f'loaded {len(train_jsons)} training json lines and {len(dev_jsons)} dev json lines.')

    if preprocess:
        train_jsons = clean_jsons(train_jsons)
        dev_jsons =  clean_jsons(dev_jsons)
    
    if add_author_name or add_author_bio:
        train_jsons = add_author_info(train_jsons, preprocess, add_author_bio, add_author_name, os.path.join(root_path, author_info_file_train))
        dev_jsons = add_author_info(dev_jsons, preprocess, add_author_bio, add_author_name, os.path.join(root_path, author_info_file_dev))
            
            
    return (train_jsons, dev_jsons)