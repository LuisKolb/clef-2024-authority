import json
import logging
import logging.handlers
import os
from clef.utils.logging_setup import set_exp_logger, setup_logging
from clef.pipeline.pipeline import step_retrieval, step_verification
from clef.utils.data_loading import AuredDataset, write_jsonlines_from_dicts
from clef.utils.scoring import eval_run_custom
from clef.verification.verify import Judge, predict_evidence

def find_best_config_str(exp_path, mode='retrieval', score_by='MAP'):
    configs = []
    # Recursively find all files named log.txt
    for root, _, files in os.walk(exp_path):
        for file in files:
            if file.endswith('log.txt'):
                # Read the file and extract configurations and scores
                with open(os.path.join(root, file)) as f:
                    for line in f:
                        if f'result for {mode} run' in line:
                            score = float(line.split(f'{score_by}: ')[1].split(' ')[0])
                            config_str = '{' + line.split('{')[1].split('}')[0] + '}'
                            config_str = config_str.replace("'", '"').replace(': True', ': "True"').replace(': False', ': "False"')
                            config = json.loads(config_str)
                            configs.append((score, config))
    # Sort configurations by score in descending order
    configs.sort(key=lambda x: x[0], reverse=True)
    return configs

def find_best_config_fp(exp_path, mode='retrieval', score_by='MAP'):
    configs = []
    # Recursively find all files named log.txt
    for root, _, files in os.walk(exp_path):
        for file in files:
            if file.endswith('log.txt'):
                # Read the file and extract configurations and scores
                with open(os.path.join(root, file)) as f:
                    for line in f:
                        if f'result for {mode} run' in line:
                            score = float(line.split(f'{score_by}: ')[1].split(' ')[0])
                            config_path = os.path.join(f.name, '..', '..', 'config.json')
                            with open(config_path) as config_file:
                                config = json.load(config_file)
                            configs.append((score, config))
    # Sort configurations by score in descending order
    configs.sort(key=lambda x: x[0], reverse=True)
    return configs

# Define parameters
root_path = '../../'  # Path to repository root
split = 'dev'
experiment_base_path = f'./experiments-data/split-{split}' 
json_data_filepath = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data', f'English_{split}.json')
golden_labels_file = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data', f'{split}_qrels.txt')

# Create retrieval configurations
configs_retrieval = []
for retriever in ['TERRIER', 'OPENAI']:
    for preprocess in [True, False]:
        for add_author_name in [True, False]:
            for add_author_bio in [True, False]:
                fingerprint = f'{"pre" if preprocess else "nopre"}-{"name" if add_author_name else "noname"}-{"bio" if add_author_bio else "nobio"}'
                config = {
                    'blind_run': False,
                    'split': split,
                    'preprocess': preprocess,
                    'add_author_name': add_author_name,
                    'add_author_bio': add_author_bio,
                    'retriever_label': retriever,
                    'retriever_k': 5,
                    'out_dir': f'{experiment_base_path}/retrieval/{retriever}/{fingerprint}',
                    'fingerprint_r': fingerprint,
                }

                # Ensure output directories exist
                os.makedirs(os.path.join(config['out_dir'], 'eval'), exist_ok=True)
                
                configs_retrieval.append(config)

# Create verification configurations
configs_verification = []
for verifier_label in ['LLAMA', 'OPENAI']:
    for preprocess in [True, False]:
        for external_data in [True, False]:
            fingerprint = f'{"pre" if preprocess else "nopre"}-{"ext" if external_data else "noext"}'
            config = {
                'blind_run': False,
                'split': split,
                'preprocess': preprocess,
                'add_author_name': external_data,
                'add_author_bio': external_data,
                'verifier_label': verifier_label,
                'out_dir': f'{experiment_base_path}/verification/{verifier_label}/{fingerprint}',
                'fingerprint_v': fingerprint,
            }

            # Ensure output directories exist
            os.makedirs(os.path.join(config['out_dir']), exist_ok=True)
            
            configs_verification.append(config)

# Create judge configurations
configs_judge = []
for norm in [True, False]:
    for scale in [True, False]:
        for ignore_nei in [True, False]:
            fingerprint = f'{"norm" if norm else "nonorm"}-{"scale" if scale else "noscale"}-{"ignore_nei" if ignore_nei else "noignore_nei"}'
            config = {
                'normalize_scores': norm,
                'scale': scale, 
                'ignore_nei': ignore_nei,
                'fingerprint_j': fingerprint,
            }
            configs_judge.append(config)

if __name__ == '__main__':
    set_exp_logger()
    logger_experiment = logging.getLogger('clef.experiment')
    
    skip_retrieval = True # set to True to skip retrieval steps
    if not skip_retrieval:
        # Execute retrieval steps
        for config in configs_retrieval:
            with open(os.path.join(config['out_dir'], 'config.json'), 'w') as f:
                f.write(json.dumps(config, indent=4))

            setup_logging(config['out_dir'])
            ds = AuredDataset(json_data_filepath, **config)
            step_retrieval(ds=ds, config=config, golden_labels_file=golden_labels_file)
        

    # Find best configuration for retrieval per method, and use that to run the verification experiments for each method
    mode = 'retrieval'
    score_by = 'MAP'
    for retriever in ['TERRIER', 'OPENAI']:
        logger_experiment.info(f'Looking for best config for {mode} with {retriever}...') 
        configs = find_best_config_fp(f'{experiment_base_path}/retrieval/{retriever}', mode, score_by)
        for i, (score, config) in enumerate(configs):
            logger_experiment.info(f'{mode} {retriever} - {i+1}. {score_by} score: {score:.4f}, config: {config}')

        _, best_retrieval_config = configs[0] # get the best config for the retrieval method
        logger_experiment.info(f'Best config for {mode} with {retriever}: {best_retrieval_config}')

        # set the trec filepath we will be using for the verification
        trec_filepath = f'{best_retrieval_config["out_dir"]}/{best_retrieval_config["retriever_label"]}-{best_retrieval_config["split"]}.trec.txt'

        # Execute verification steps
        for verifier_config_ref in configs_verification:
            verifier_config = verifier_config_ref.copy() # copy the config so we can modify it
            verifier_config['retriever_label'] = best_retrieval_config['retriever_label']
            verifier_config['fingerprint_r'] = best_retrieval_config['fingerprint_r']
            verifier_config['out_dir'] = f'{verifier_config["out_dir"]}/ret-{best_retrieval_config["retriever_label"]}'

            # if out_dir exists, skip this (useful in case of partial execuotion if the script is interrupted)
            if os.path.exists(verifier_config['out_dir']):
                continue


            os.makedirs(verifier_config['out_dir'], exist_ok=True) 
            setup_logging(verifier_config['out_dir'])

            # instantiate the verifier
            if 'LLAMA' in verifier_config["verifier_label"].upper():
                from clef.verification.models.hf_llama3 import Llama3Verifier
                verifier = Llama3Verifier()
            elif 'OPENAI' in verifier_config["verifier_label"].upper():
                from clef.verification.models.open_ai import OpenaiVerifier
                verifier = OpenaiVerifier()

            # create the dataset, add the judgements, and run the verification with the chosen verifier
            # only do this once since we can reuse the predictions as we only want to test the judge later
            ds = AuredDataset(json_data_filepath, **verifier_config)
            
            ds.add_trec_file_judgements(trec_filepath, sep=' ', normalize_scores=False) # normalizing scores has no influence here, as its only later used in judge...
            verification_decisions = predict_evidence(ds, verifier)

            # loop over the different judge configurations to find the best one
            for judge_config in configs_judge:
                # combine the base config with the judge config
                verification_config = {**verifier_config, **judge_config}

                # need to recreate the data set since we are testing the normalization of retriever scores, ...
                # ...which is added when trec file is added to the dataset
                ds = AuredDataset(json_data_filepath, **verification_config)
                ds.add_trec_file_judgements(trec_filepath, sep=' ', normalize_scores=verification_config['normalize_scores'])

                solomon = Judge(scale=verification_config['scale'], ignore_nei=verification_config['ignore_nei'])
                ds_grouped = ds.get_grouped_rumors()
                res_jsons = []

                # loop over the rumors in the dataset, and combine the scores using the judge config
                for rumor_id in ds_grouped:
                    evidence_predictions = []
                    claim = ds_grouped[rumor_id]['rumor']
                    label = ds_grouped[rumor_id]['label']
                    retrieved_evidence = ds_grouped[rumor_id]['retrieved_evidence']
                    
                    for post_id, prediction in verification_decisions[rumor_id]:
                        post = next((p for p in retrieved_evidence if p.post_id == post_id), None)
                        evidence_predictions.append((claim, post, prediction))

                    pred_label, pred_evidence = solomon(evidence_predictions)

                    predicted = {
                        "id": rumor_id,
                        "label": label,
                        "claim": claim,
                        "predicted_label": pred_label,
                        "predicted_evidence": pred_evidence,
                    }
                    res_jsons.append(predicted)
                
                # make the dirs to save experiment info to
                verification_config['out_dir'] = f'{verification_config["out_dir"]}/{verification_config["fingerprint_j"]}'
                os.makedirs(verification_config['out_dir'], exist_ok=True)

                # save the verification results to a jsonl file
                verification_outfile = f'{verification_config["out_dir"]}/zeroshot-ver-{verification_config["verifier_label"]}-retr-{verification_config["retriever_label"]}.jsonl'
                write_jsonlines_from_dicts(verification_outfile, res_jsons)

                # score the verification results
                macro_f1, strict_macro_f1 = eval_run_custom(verification_outfile, json_data_filepath, '')
                
                # save the config used for the verification run
                with open(os.path.join(verification_config['out_dir'], 'config.json'), 'w') as f:
                    f.write(json.dumps(verification_config, indent=4))

                # write the result summary both to the logger and save it to a file
                from clef.pipeline.pipeline import logger
                logger.info(f'Result for verification run - Macro-F1: {macro_f1:.4f}, Strict-Macro-F1: {strict_macro_f1:.4f} with config {verification_config} and TREC FILE {trec_filepath}')
                
                with open(os.path.join(verification_config['out_dir'], 'log.txt'), 'a') as fh:
                    fh.write(f'Result for verification run - Macro-F1: {macro_f1:.4f}, Strict-Macro-F1: {strict_macro_f1:.4f} with config {verification_config} and TREC FILE {trec_filepath}\n')

    #
    # this part is optional... the results of the experiment are more convenient to analyze in a jupyter notebook
    #

    print_best_configs = False
    if print_best_configs:
        # Find best configuration for verification overall
        mode = 'verification'
        score_by = 'Macro-F1'
        logger_experiment.info(f'Looking for best config for {mode}...') 
        configs = find_best_config_fp(f'{experiment_base_path}/{mode}', mode, score_by)
        for i, (score, config) in enumerate(configs):
            logger_experiment.info(f'{mode} - {i+1}. {score_by} score: {score:.4f}, config: {config}')
        
        # Separate and log configurations with and without external data
        try:
            no_ext_data = [item for item in configs if not (item[1]['add_author_name'] or item[1]['add_author_bio'])]
            ext_data = [item for item in configs if item[1]['add_author_name'] or item[1]['add_author_bio']]

            no_ext_data.sort(key=lambda x: x[0], reverse=True)
            ext_data.sort(key=lambda x: x[0], reverse=True)

            for i, (score, config) in enumerate(no_ext_data):
                logger_experiment.info(f'{mode} no ext data - {i+1}. {score_by} score: {score:.4f}, config: {config["out_dir"]}/{config["fingerprint_j"]}/config.json- {config}')

            for i, (score, config) in enumerate(ext_data):
                logger_experiment.info(f'{mode} using ext data - {i+1}. {score_by} score: {score:.4f}, config: {config["out_dir"]}/{config["fingerprint_j"]}/config.json - {config}')    

        except Exception as e:
            logger_experiment.error(f'Error processing configurations: {e}')
            raise e
