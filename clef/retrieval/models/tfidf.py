from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from clef.retrieval.retrieve import EvidenceRetriever

def retrieve_relevant_documents_tfidf(rumor_id, query, timeline, k=5):
    # Get only doc texts
    documents = [t[2] for t in timeline]
    tweet_ids = [t[1] for t in timeline]

    # Combine query and documents for TF-IDF vectorization
    combined_texts = [query] + documents

    # Generate TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Calculate similarity of the query to each document
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]) # type: ignore
    
    # Rank documents based on similarity scores
    ranked_doc_indices = similarity_scores.argsort()[0][::-1]

    ranked = []
    for i, idx in enumerate(ranked_doc_indices[:k]):
        ranked += [[rumor_id, tweet_ids[idx], i, similarity_scores[0][idx]]]
    
    return ranked

    # # Sort the documents according to rank
    # ranked_documents = [documents[i] for i in ranked_doc_indices]
    # ranked_scores = [similarity_scores[0][i] for i in ranked_doc_indices]
    # ranked_ids = [tweet_ids[i] for i in ranked_doc_indices]

    # # Create a list of tuples of shape (doc, score)
    # ranked_tuples = (list(zip(ranked_ids, ranked_scores, ranked_documents)))
    
    # return ranked_tuples

class TFIDFRetriever(EvidenceRetriever):
    def __init__(self, k):
        super().__init__(k)

    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs):
        # Get only doc texts
        documents = [tweet[2] for tweet in timeline]
        tweet_ids = [tweet[1] for tweet in timeline]

        # Combine query and documents for TF-IDF vectorization
        combined_texts = [claim] + documents

        # Generate TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        # Calculate similarity of the query to each document
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]) # type: ignore

        # Rank documents based on similarity scores
        ranked_doc_indices = similarity_scores.argsort()[0][::-1]

        ranked = []
        for i, idx in enumerate(ranked_doc_indices[:self.k]):
            ranked.append([rumor_id, tweet_ids[idx], i, similarity_scores[0][idx]])

        return ranked