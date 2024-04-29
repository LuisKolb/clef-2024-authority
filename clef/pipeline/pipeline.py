#
# imports
#

import os
from typing import Optional
from clef.utils.data_loading import AuredDataset, write_jsonlines_from_dicts
from clef.utils.data_loading import write_trec_format_output
from clef.utils.scoring import eval_run_custom
from clef.verification.models.open_ai import OpenaiVerifier
from clef.verification.verify import Judge, run_verifier_on_dataset


import logging
# logging to file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./pipeline.log',
                    filemode='w')

# logging to console
console = logging.StreamHandler() # define a Handler which writes INFO messages or higher to the sys.stderr
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s') # set a format which is simpler for console use
console.setFormatter(formatter) # tell the handler to use this format
logging.getLogger().addHandler(console) # add the handler to the root logger

logger_pipe = logging.getLogger('clef.pipe')

def step_retrieval(ds: AuredDataset, config, golden_labels_file=None):
    # ensure out_dir directories exist for saving output (required for anserini, etc - not only for eval)
    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
        if not os.path.exists(os.path.join(config['out_dir'], 'eval')):
            os.makedirs(os.path.join(config['out_dir'], 'eval'))

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

    elif 'TERRIER' in config['retriever_label'].upper():
        from clef.retrieval.models.terrier import TerrierRetriever
        retriever = TerrierRetriever(config['retriever_k'])

    else:
        logger_pipe.error(f"retriever type {config['retriever_label']} not valid!")
        quit()

    from clef.retrieval.retrieve import retrieve_evidence
    data = retrieve_evidence(ds, retriever)

    trec_filepath = f'{config["out_dir"]}/{config["retriever_label"]}-{config["split"]}.trec.txt'
    write_trec_format_output(trec_filepath, data, config['retriever_label'])

    if golden_labels_file:
        # gold labels available
        from clef.utils.scoring import eval_run_retrieval
        r5, meanap = [v for v in eval_run_retrieval(trec_filepath, golden_labels_file).values()]

        logger_pipe.info(f'result for retrieval run - R@5: {r5:.4f} MAP: {meanap:.4f} with config {config}')
        with open(os.path.join(config['out_dir'], 'eval', 'log.txt'), 'a') as fh:
            fh.write(f'result for retrieval run - R@5: {r5:.4f} MAP: {meanap:.4f} with config {config}\n')
        return r5, meanap
    else:
        # running blind
        return trec_filepath


def step_verification(ds: AuredDataset, config,  ground_truth_filepath=None):
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
    trec_filepath = f'{config["out_dir"]}/{config["retriever_label"]}-{config["split"]}.trec.txt'
    ds.add_trec_file_judgements(trec_filepath, sep=' ',
                                normalize_scores=config['normalize_scores'])

    verifier = OpenaiVerifier()

    solomon = Judge(scale=config['scale'], 
                    ignore_nei=config['ignore_nei'])
    
    verification_results = run_verifier_on_dataset(ds, verifier, solomon)

    verification_outfile = f'{config["out_dir"]}/zeroshot-ver-rq3-openai.jsonl'
    write_jsonlines_from_dicts(verification_outfile, verification_results)

    # only run evaluation scoring if not running blind
    if  ground_truth_filepath:
        # gold labels available
        macro_f1, sctrict_macro_f1 = eval_run_custom(verification_outfile, ground_truth_filepath, '')

        logger_pipe.info(f'result for verification run - Strict-F1: {macro_f1:.4f} Strict-Macro-F1: {sctrict_macro_f1:.4f} with config {config} and TREC FILE {trec_filepath}')
        with open(os.path.join(config['out_dir'], 'eval', 'log.txt'), 'a') as fh:
            fh.write(f'result for verification run - Strict-F1: {macro_f1:.4f} Strict-Macro-F1: {sctrict_macro_f1:.4f} with config {config} and TREC FILE {trec_filepath}\n')

        return verification_results
    else:
        # running blind
        return verification_outfile


def run_pipeline(data_path: str, root_path: str, task5_dir: str, config: dict):
    if config["split"] == "train":
        json_data_filepath = os.path.join(data_path, 'English_train.json') # relative to root
        golden_labels_file = os.path.join(root_path, 'clef', 'data', 'train_qrels.txt') # relative to root, was generated by luis

    elif config["split"] == "dev":
        json_data_filepath = os.path.join(data_path, 'English_dev.json') # relative to root
        golden_labels_file = os.path.join(root_path, task5_dir, 'data', 'dev_qrels.txt') # relative to root

    """
    step 0: load data from file and (optionally) preprocess data
    """
    ds = AuredDataset(json_data_filepath, **config)
    step_retrieval(ds, golden_labels_file, config)
    step_verification(ds, json_data_filepath, config)


def main(config: Optional[dict]):
    #
    # define default parameters
    #
    root_path = '../../' # path to github repository root level (where setup.py is located)

    task5_dir = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5') # relative to root

    data_path = os.path.join(root_path, task5_dir, 'data') # relative to root

    if not config:
        # default config
        config = {
            'split': 'dev',
            'preprocess': True,
            'add_author_name': False,
            'add_author_bio': False,
            'out_dir': './data-out/pipeline-default',
            'retriever_k': 5,
            'retriever_label': 'TERRIER',
            'normalize_scores': True,
            'scale': False, 
            'ignore_nei': True,
        }

    logger_pipe.info(f"Running Pipeline with default parameters: {config}")

    run_pipeline(data_path, root_path, task5_dir, config)


if __name__ == "__main__":
    main(config=None)
