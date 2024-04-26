from typing import Dict, List
from tqdm.auto import tqdm

from clef.utils.preprocessing import clean_tweet_aggressive

def retrieve_evidence_old(dataset: List, method: str, kwargs: Dict = {}):
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

from typing import Dict, List
from abc import ABC, abstractmethod
from clef.utils.data_loading import AuredDataset

import logging
logger_retrieval = logging.getLogger('clef.retrv')

# Base class for retrieval
class EvidenceRetriever(ABC):
    def __init__(self, k):
        self.k = k
        
    @abstractmethod
    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs) -> List:
        """Retrieve documents based on the input parameters."""
        pass


# Specific retriever subclasses
def retrieve_evidence(dataset: AuredDataset, retriever: EvidenceRetriever, kwargs: Dict = {}):
    data = []

    for i, item in enumerate(dataset):
        rumor_id = item["id"]
        claim = item["rumor"]
        timeline = item["timeline"]
        logger_retrieval.info(f"({i}/{len(dataset)}) Retrieving data for rumor_id {rumor_id} using {retriever.__class__}")

        retrieved_data = retriever.retrieve(rumor_id, claim, timeline, **kwargs)
        data.extend(retrieved_data)
        logger_retrieval.debug(f"Retrieved data: {retrieved_data}")

    return data
