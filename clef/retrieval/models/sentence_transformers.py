from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_documents_sbert(rumor_id, query, timeline, k=5):
    corpus = [t[2] for t in timeline]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    top_k = min(k, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    # if debug:
    #     print("\n\n======================\n\n")
    #     print("Query:", query)
    #     evidence_ids = [e[1] for e in evidence]

    found = []
    docs = []

    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            id = timeline[idx][1]

            # if debug:
            #     is_evidence = id in evidence_ids
            #     star = "(*)" if is_evidence else "\t"
            #     print(star, '\t', "(Rank: {:.0f})".format(i+1), "(Score: {:.4f})".format(score), corpus[idx])
            #     if is_evidence: found += [id]

            docs += [[rumor_id, id, i+1, score.item()]]

    # if debug:    
    #     for _, ev_id, ev_text in evidence:
    #         if ev_id not in found:
    #                 print('(!) ', ev_text)
    
    return docs