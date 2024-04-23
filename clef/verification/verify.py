from typing import Callable, List
from tqdm.auto import tqdm
from clef.utils.data_loading import RankedDocs
import re


def factcheck_using_evidence(claim: str, evidence: List[RankedDocs], inference_method: Callable, debug: bool = False, model_string: str = ''):
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


from clef.utils.preprocessing import clean_tweet_aggressive
from tqdm.auto import tqdm


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
    elif model == 'llama3':
        from clef.verification.models.ollama import inference_llama3
        inference_method = inference_llama3
    else:
        print('[ERROR] invalid model string, available options: ["bart"|"roberta"|"openai"|"llama3"]')
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