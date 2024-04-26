

import logging
logger_pipe = logging.getLogger('clef.pipe')


#
# define parameters
#
from clef.utils.data_loading import AuredDataset, task5_dir
from clef.utils.data_loading import write_trec_format_output
import os

def run_pipe(filepath, config, golden_labels_file):


    # ensure out_dir directories exist for saving output (required for anserini, etc - not only for eval)
    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
        if not os.path.exists(os.path.join(config['out_dir'], 'eval')):
            os.makedirs(os.path.join(config['out_dir'], 'eval'))

    """
    step 0: load data from file and (optionally) preprocess data
    """

    ds = AuredDataset(filepath, **config)

    """
    step 1 output format:

    rumor_id Q0 authority_tweet_id rank score tag

    where

        rumor_id: The unique ID for the given rumor
        Q0: This column is needed to comply with the TREC format
        authority_tweet_id: The unique ID for the authority tweet ID
        rank: the rank of the authority tweet ID for that given rumor_id
        score: the score given by your model for the authority tweet ID given the rumor
        tag: the string identifier of the team/model


    needs input at least:
    - rumor id
    - claim
    - timeline (authority tweets)
    - config params for the retrieval component

    The official evaluation measure for evidence retrieval is Mean Average Precision (MAP). 
    The systems get no credit if they retrieve any tweets for unverifiable rumors. 
    Other evaluation measures to be considered are Recall@5.
    """
    if 'LUCENE' in  config['retriever_label'].upper():
        from clef.retrieval.models.pyserini import LuceneRetriever
        retriever = LuceneRetriever(config['retriever_k'])

    elif 'OPENAI' in config['retriever_label'].upper():
        from clef.retrieval.models.openai import OpenAIRetriever
        retriever = OpenAIRetriever(config['retriever_k'])

    elif 'SBERT' in config['retriever_label'].upper():
        from clef.retrieval.models.sentence_transformers import SBERTRetriever
        retriever = SBERTRetriever(config['retriever_k'])

    elif 'TFIDF' in config['retriever_label'].upper():
        from clef.retrieval.models.tfidf import TFIDFRetriever
        retriever = TFIDFRetriever(config['retriever_k'])

    else:
        logger_pipe.error(f"retriever {config['retriever_label']} not valid!")
        quit()

    from clef.retrieval.retrieve import retrieve_evidence
    data = retrieve_evidence(ds, retriever)

    trecfile = f'{config["out_dir"]}/{config["retriever_label"]}-dev.trec.txt'
    write_trec_format_output(trecfile, data, config['retriever_label'])

    from clef.utils.scoring import eval_run_retrieval
    r5, meanap = [v for v in eval_run_retrieval(trecfile, golden_labels_file).values()]

    logger_pipe.info(f'result for retrieval run - R@5: {r5:.4f} MAP: {meanap:.4f} with config {config}')
    with open(os.path.join(config['out_dir'], 'eval', 'log.txt'), 'a') as fh:
        fh.write(f'result for retrieval run - R@5: {r5:.4f} MAP: {meanap:.4f} with config {config}\n')

    """
    step 2 output format:

    {
        "id": rumor_id,
        "predicted_label": rumor label as predicted by your model, 
        "predicted_evidence":[
            [authority, authority tweet ID, authority tweet text, score as predicted by your model],
            [authority, authority tweet ID, authority tweet text, score as predicted by your model],
            [authority, authority tweet ID, authority tweet text, score as predicted by your model], 
            ...
        ]
    }

    needs input at least:
    - rumor id
    - claim
    - ranked evidence (authority tweets)
    - config params for the verification component


    We use the Macro-F1 to evaluate the classification of the rumors. 
    Additionally, we will consider a Strict Macro-F1 where the rumor label is considered correct only if at least one retrieved authority evidence is correct.
    """
    return