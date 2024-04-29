from typing import Generator, List, Tuple, Dict, TypedDict, Union, Optional, NamedTuple
import json
import os
import re
from clef.utils.preprocessing import clean_text_basic

import logging
logger_loading = logging.getLogger('clef.loader')

class AuthorityPost(NamedTuple):
    url: str
    post_id: str
    text: str
    rank: Optional[int]
    score: Optional[float]


class RumorWithEvidence(TypedDict):
    id: str
    rumor: str
    label: str
    timeline: List[AuthorityPost]
    evidence: List[AuthorityPost]
    retrieved_evidence: Optional[List[AuthorityPost]] # not required


class AuredDataset(object):
    def __init__(self, filepath, preprocess, add_author_name, add_author_bio, **kwargs) -> None:
        self.filepath: Union[str, os.PathLike] = filepath
        self.rumors: List[RumorWithEvidence] = []

        """
        init ds like this (for example):

        config = {
            'preprocess': True,
            'add_author_name': False,
            'add_author_bio': False,
            ...
        }
        ds = AuredDataset(filepath, **config)
        """
        self.preprocess: bool = preprocess
        self.add_author_name: bool = add_author_name
        self.add_author_bio: bool = add_author_bio

        self.load_rumor_data()


    def __str__(self) -> str:
        return json.dumps(self.rumors, indent=2)
    
    def __iter__(self) -> Generator[RumorWithEvidence,None,None]:
        for rumor_item in self.rumors:
            yield rumor_item
    
    def __getitem__(self, idx):
        return self.rumors[idx]

    def __setitem__(self, idx, val):
        self.rumors[idx] = val
    
    def __len__(self) -> int:
        return len(self.rumors)
    
    def load_rumor_data(self):
        jsons = self.load_rumors_from_jsonl()

        for item in jsons:
            entry = RumorWithEvidence(item)
            entry['timeline'] = [AuthorityPost(*post, None, None) for post in entry['timeline']] # type: ignore
            entry['evidence'] = [AuthorityPost(*post, None, None) for post in entry['evidence']] # type: ignore
            entry['retrieved_evidence'] = None
            self.rumors.append(entry)

        logger_loading.info(f'loaded {len(jsons)} json entries from {self.filepath}')

        for item in self.rumors:
            item['timeline'] = self.format_posts(item['timeline'])
            item['evidence'] = self.format_posts(item['evidence'])
            if self.preprocess:
                item['rumor'] = clean_text_basic(item['rumor'])
    
    def get_grouped_rumors(self):
        """
        returns a dict with mapping {rumor_id: RumorWithEvidence}
        """
        grouped = {} 
        for item in self.rumors:
            grouped[item['id']] = item # add to grouped dict 
        return grouped

    def load_rumors_from_jsonl(self) -> List[RumorWithEvidence]:
        jsons = []
        with open(self.filepath, encoding='utf-8') as file:
            for line in file:
                jsons += [json.loads(line)]
        return jsons
    
    def format_posts(self, post_list: List[AuthorityPost], author_info_filepath: str = '../../clef/data/author-data-translated.json'):
        new_post_list = []
        author_info = {}
        if self.add_author_bio or self.add_author_name:
            with open(author_info_filepath, 'r') as file:
                author_info = json.load(file)

        for post in post_list:
            new_post_text = f'AuthorityStatement: "{post.text}"'

            if author_info:
                account = post.url.strip()
                name = author_info[account]["translated_name"]
                bio = author_info[account]["translated_bio"]
                
                if self.add_author_bio:
                    new_post_text = f'AuthorityDescription: {bio}\n' + new_post_text
                if self.add_author_name:
                    new_post_text = f'AuthorityName: {name}\n' + new_post_text

            if self.preprocess:
                new_post_text = clean_text_basic(new_post_text)
            
            new_post_list.append(AuthorityPost(post.url, post.post_id, new_post_text, None, None))
        return new_post_list
    
    def add_trec_file_judgements(self, trec_judgements_path, sep=' ', normalize_scores=True):
        """
        create a list of RankedDocs objects in key retrieved_evidence from TREC-formatted file

        Parameters:
            - trec_judgements_path: filepath to TREC-formatted file containing rank and score
            - sep: separator used in TREC file

        Returns:
        list of json-like rumors from dataset, with the key retrieved_evidence populated with a list of RankedDocs-like objects
        """
        trec_by_id = {}
        max_score = 0
        min_score = 0
        num_rows = 0

        with open(trec_judgements_path, 'r') as file:
            for line in file:
                # handle edge case where dataset may contain field which contain an additional whitespace, ...
                # which was then saved together with the field value as a string looking like 'id 0 "id " 1'
                # (only seems to affect TERRIER trec files)
                line = re.sub('"', '', line)
                line = re.sub('  ', ' ', line)
                rumor_id, _, evidence_id, rank, score, tag = line.split(sep)
                score = float(score)
                if rumor_id not in trec_by_id:
                    trec_by_id[rumor_id] = {}
                
                # keep max,min score values to normalize later
                # scores should always be positive, but still do this...
                if score < min_score:
                    min_score = score
                if score > max_score:
                    max_score = score

                # add entry to dict for lookup later 
                trec_by_id[rumor_id][evidence_id] = (rank, score)
                num_rows += 1
        
        if (max_score-min_score) == 0:
            logger_loading.error(f'encountered (max_score-min_score) == 0; max={max_score}; min={min_score}')
            raise ValueError()

        for i, item in enumerate(self.rumors):
            timeline: List[AuthorityPost] = item['timeline']
            
            item['retrieved_evidence'] = []

            doc_ranks = trec_by_id[item['id']]

            for post in timeline:
                if post.post_id in doc_ranks:
                    rank, score = doc_ranks[post.post_id]
                    
                    # normalize score to [0...1] using max,min scores from earlier
                    if not normalize_scores: 
                        score_norm = score
                    else: 
                        score_norm = (score-min_score) / (max_score-min_score)
                    
                    item['retrieved_evidence'].append(AuthorityPost(
                        post.url, 
                        post.post_id, 
                        post.text,
                        int(rank),
                        float(score_norm),
                    )) 
            
            self.rumors[i] = item
        
        logger_loading.info(f'added {num_rows} scores from {trec_judgements_path} to the evidence entries')

#
# OLD STUFF
#

def load_rumors_from_jsonl(filepath: Union[str, os.PathLike]) -> List[RumorWithEvidence]:
    jsons = []
    with open(filepath, encoding='utf8') as file:
        for line in file:
            jsons += [json.loads(line)]
    return jsons


def write_trec_format_output(filename: str, data: List[List[Union[str, int, float]]], tag: str = "NO_TAG_SPECIFIED") -> None:
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
                # use " " as separator, pyterrier uses " " as separator by default!! (stupid, please just use "\t")
                line = f"{rumor_id} Q0 {authority_tweet_id} {rank} {score} {tag}\n"
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