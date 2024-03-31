from typing import Dict, List
from tqdm.auto import tqdm

from clef.utils.preprocessing import clean_tweet_aggressive

def retrieve_evidence(dataset: List, method: str, kwargs: Dict = {}):
    method = method.upper()
    
    if method == 'LUCENE':
        from clef.retrieval.models.pyserini import searchPyserini
        search = searchPyserini
    elif method == 'TFIDF':
        from clef.retrieval.models.tfidf import retrieve_relevant_documents_tfidf
        search = retrieve_relevant_documents_tfidf
    elif method == 'SBERT':
        from clef.retrieval.models.sentence_transformers import retrieve_relevant_documents_sbert
        search = retrieve_relevant_documents_sbert
    elif method == 'OPENAI':
        from clef.retrieval.models.openai import retrieve_relevant_documents_openai
        search = retrieve_relevant_documents_openai
    else:
        print(f'[ERROR] method "{method}" not known')
        return []

    data = []

    for item in tqdm(dataset):
        rumor_id = item['id']
        claim = item['rumor']
        timeline = item['timeline']

        data += search(rumor_id, claim, timeline, **kwargs)

    return data