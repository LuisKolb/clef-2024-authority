import re
from typing import Callable, List, NamedTuple, Tuple
from tqdm.auto import tqdm
from clef.utils.data_loading import AuredDataset, AuthorityPost
from clef.verification.models.openai import BaseVerifier

import logging
logger_verification = logging.getLogger('clef.retr')

class VerificationResult(NamedTuple):
    label: str
    score: float

class Judge(object):
    scale: bool
    ignore_nei: bool
    threshold_refutes: float
    threshold_supports: float
    
    def __init__(self, scale=False, ignore_nei=True, threshold_refutes=0.15, threshold_supports=-0.15) -> None:
        self.scale = scale
        self.ignore_nei = ignore_nei
        self.threshold_refutes = threshold_refutes
        self.threshold_supports = threshold_supports

    def __call__(self, evidence_predictions: List[Tuple[str,AuthorityPost,VerificationResult]]) -> Tuple:
        return self.judge_evidence(evidence_predictions)
    
    def judge_evidence(self, evidence_predictions: List[Tuple[str,AuthorityPost,VerificationResult]]) -> Tuple:
        """
        take in claim, evidence and decision from verifier, and return the **overall** label we'll predict for the rumor

        CLEF CheckThat! task 5: score is [-1, +1] where 
        +1 means evidence strongly refutes
        -1 means evidence strongly supports
        """

        confidences = []
        predicted_evidence = []

        for claim, post, prediction in evidence_predictions:
            # if self.ignore_nei and prediction.label == "NOT ENOUGH INFO":
            #     continue

            confidence = float(prediction.score)

            # predicted confidence from verifier should always be positive
            if prediction.label == "SUPPORTS" and confidence > 0:
                # for SUPPORTS we'll flip confidence negative
                confidence *= -1
            elif prediction.label == "NOT ENOUGH INFO":
                confidence *= 0 # TODO should NEI contribute to judgement? probably not
            elif prediction.label == "REFUTES" and confidence < 0:
                # in case confidence isn't positive, flip it to positive
                confidence *= -1

            if self.scale and post.score:
                confidence = confidence * post.score # scale by retrieval score

            if not prediction.label == "NOT ENOUGH INFO":
                confidences += [confidence]
            elif self.ignore_nei:
                # prediction is NEI, skip for label calculation
                pass
            else:
                # prediction is NEI, include in label calculation
                confidences += [confidence]

            # build return list for submission
            predicted_evidence.append([
                post.url,
                post.post_id,
                post.text,
                confidence,
            ])


        if confidences:
            # mean confidence, no weighting
            meanconf = sum(confidences) / len(confidences)

            if meanconf > self.threshold_refutes:
                pred_label = "REFUTES"
            elif meanconf < self.threshold_supports:
                pred_label = "SUPPORTS"
            else:
                # evidence not conclusive enough
                pred_label = "NOT ENOUGH INFO"
        else:
            # no relevant judgements for the given evidence
            pred_label = "NOT ENOUGH INFO"

        logger_verification.debug(f'judged {pred_label} for confidences {confidences}')

        return pred_label, predicted_evidence


def judge_using_evidence(rumor_id, claim: str, evidence: List[AuthorityPost], verifier: BaseVerifier, judge: Judge):
    evidences_with_decisions = []
    

    for post in evidence:
        if not post.text:
            logger_verification.warn(f'evidence text empty for rumor with id {rumor_id}; evidence={post}')
            continue

        prediction = verifier(claim, post.text)
        evidences_with_decisions.append((claim,post,prediction))

    return  judge(evidences_with_decisions)
    

def run_verifier_on_dataset(dataset: AuredDataset, verifier: BaseVerifier, judge: Judge) -> List:
    res_jsons = []

    for i, item in enumerate(dataset):
        rumor_id = item["id"]
        label = item["label"]
        claim = item["rumor"]

        if not item["retrieved_evidence"]:
            # only run fact check if we actually have retrieved evidence
            logger_verification.warn(f'key "retrieved_evidence" was empty for rumor with id {claim}')
            return []
        
        retrieved_evidence = item["retrieved_evidence"] 
        
        pred_label, pred_evidence = judge_using_evidence(rumor_id, claim, retrieved_evidence, verifier, judge)

        print(f'({i}/{len(dataset)}) Verifying {rumor_id}: {claim}')

        for url, post_id, text, confidence in pred_evidence:
            formatted_text = re.sub(r"\s+", " ", text) # replace linebreaks, etc. for pretty printing in a single line
            print(f'\t{confidence} {formatted_text}')

        print(f'label:\t\t{label}')
        print(f'predicted:\t{pred_label}')

        res_jsons.append(
            {
                "id": rumor_id,
                "label": label,
                "claim": claim,
                "predicted_label": pred_label,
                "predicted_evidence": pred_evidence,
            }
        )
    
    return res_jsons

#
# OLD CODE
#


def factcheck_using_evidence(claim: str, evidence: List[AuthorityPost], inference_method: Callable, debug: bool = False, model_string: str = ''):
    predicted_evidence = []
    confidences = []

    if debug: tqdm.write(f'{claim}')

    for author_account, tweet_id, evidence_text, rank, score in evidence:
        if not evidence_text:
            if debug: tqdm.write('[DEBUG] evidence string empty')
            return ("NOT ENOUGH INFO", [])
        if model_string:
            label, confidence = inference_method(claim, evidence_text, model_string)
        else:
            label, confidence = inference_method(claim, evidence_text)

        # CLEF CheckThat! task 5: score is [-1, +1] where 
        #   -1 means evidence strongly refutes
        #   +1 means evidence strongly supports

        # confidence = confidence * score # scale by retrieval score

        if label == "REFUTES":
            # confidence is always positive, for REFUTES make confidence negative
            confidence *= -1
        elif label == "NOT ENOUGH INFO":
            confidence *= 0 # TODO uhmmm...

        predicted_evidence += [[
            author_account,
            tweet_id,
            evidence_text,
            confidence,
        ]]

        if label != "NOT ENOUGH INFO":
            confidences += [confidence]
        if debug: 
            formatted_text = re.sub(r"\s+", " ", evidence_text)
            tqdm.write(f'\t{confidence} {formatted_text}')

    if confidences:
        meanconf = sum(confidences) / len(confidences) # mean confidence, no weighting
    else:
        meanconf = 0
    
    if meanconf > 0.1:
        pred_label = "SUPPORTS"
    elif meanconf < -0.1:
        pred_label = "REFUTES"
    else:
        pred_label = "NOT ENOUGH INFO"
    
    return pred_label, predicted_evidence



def check_dataset_with_model(dataset: List, model: str, debug: bool = False, model_string: str = '') -> List:
    res_jsons = []

    if model == 'bart':
        from clef.verification.models.bart import inference_bart
        inference_method = inference_bart
    elif model == 'roberta':
        from clef.verification.models.roberta import inference_roberta
        inference_method = inference_roberta
    elif model == 'openai':
        from clef.verification.models.openai import inference_openai
        inference_method = inference_openai
    elif model.startswith('ollama'):
        from clef.verification.models.ollama import inference_llama3
        inference_method = inference_llama3
    elif model.startswith('llama3'):
        from clef.verification.models.hf_llama3 import inference_hf_llama3
        inference_method = inference_hf_llama3
    else:
        print('[ERROR] invalid model string, available options: ["bart"|"roberta"|"openai"|"ollama"|"llama3"]')
        return []

    for item in tqdm(dataset):
        rumor = item["rumor"]
        retrieved_evidence = item["retrieved_evidence"]

        if retrieved_evidence: # only run fact check if we actually have retrieved evidence
            pred_label, pred_evidence = factcheck_using_evidence(rumor, retrieved_evidence, inference_method, debug, model_string)

            if debug:
                tqdm.write(f'label: {item["label"]}')
                tqdm.write(f'predicted: {pred_label}')
                tqdm.write('')
            
            res_jsons += [
                {
                    "id": item["id"],
                    "label": item["label"],
                    "claim": rumor,
                    "predicted_label": pred_label,
                    "predicted_evidence": pred_evidence,
                }
            ]
    
    return res_jsons