
import json
import logging
import logging.handlers
import os
from clef.utils.logging_setup import setup_logging 


from clef.pipeline.pipeline import step_retrieval, step_verification
from clef.utils.data_loading import AuredDataset, write_jsonlines_from_dicts
from clef.utils.scoring import eval_run_custom
from clef.verification.verify import Judge, predict_evidence

def find_best_config_str(exp_path, mode='retrieval', score_by='MAP'):
    l = []
    # find al files called log.txt, recursively
    for root, dirs, files in os.walk(exp_path):
        for file in files:
            if file.endswith('log.txt'):
                # open file, and for each line test if it contains the string "result for {mode} run - "
                with open(os.path.join(root, file)) as f:
                    for line in f:
                        if f'result for {mode} run' in line:
                            score = float(line.split(f'{score_by}: ')[1].split(' ')[0])
                            # now, get string between the brackets { and }
                            configs = '{' + line.split('{')[1].split('}')[0] + '}'
                            # replace single quotes with double quotes
                            configs = configs.replace("'", '"')
                            # convert each boolean value to string
                            configs = configs.replace(': True', ': "True"')
                            configs = configs.replace(': False', ': "False"')
                            # convert to json
                            config = json.loads(configs)
                            l.append((score, config))

    # sort the list by the first element, which is the score
    l.sort(key=lambda x: x[0], reverse=True)
    return l

def find_best_config_fp(exp_path, mode='retrieval', score_by='MAP'):
    l = []
    # find al files called log.txt, recursively
    for root, dirs, files in os.walk(exp_path):
        for file in files:
            if file.endswith('log.txt'):
                # open file, and for each line test if it contains the string "result for {mode} run - "
                with open(os.path.join(root, file)) as f:
                    for line in f:
                        if f'result for {mode} run' in line:
                            score = float(line.split(f'{score_by}: ')[1].split(' ')[0])
                            config = json.load(open(os.path.join(f.name, '..', '..', 'config.json')))
                            l.append((score, config))

    # sort the list by the first element, which is the score
    l.sort(key=lambda x: x[0], reverse=True)
    return l

#
# define parameters
#

root_path = '../../' # path to github repository root level (where setup.py is located)

json_data_filepath = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data', 'English_dev.json')

golden_labels_file = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data', 'dev_qrels.txt')

configs_retrieval = []
configs_judge = []

for retriever in ['TERRIER', 'OPENAI']:
    for preprocess in [True, False]:
        for add_author_name in [True, False]:
            for add_author_bio in [True, False]:
                fingerprint = f'{"pre" if preprocess else "nopre"}-{"name" if add_author_name else "noname"}-{"bio" if add_author_bio else "nobio"}'
                config = {
                    'blind_run': False,
                    'split': 'dev',
                    'add_author_name': add_author_name,
                    'add_author_bio': add_author_bio,
                    'retriever_k': 5,
                    'out_dir': f'./data-out/experiments/test-dev/retr-{retriever}/{fingerprint}',
                    'preprocess': preprocess,
                    'retriever_label': retriever,
                }

                # ensure out_dir directories exist for saving output (required for anserini, etc - not only for eval)
                if not os.path.exists(config['out_dir']):
                    os.makedirs(config['out_dir'])
                    if not os.path.exists(os.path.join(config['out_dir'], 'eval')):
                        os.makedirs(os.path.join(config['out_dir'], 'eval'))
                
                configs_retrieval.append(config)

# judge options
for norm in [True, False]:
    for scale in [True, False]:
        for ignore_nei in [True, False]:
            fingerprint = f'{"norm" if norm else "nonorm"}-{"scale" if scale else "noscale"}-{"ignore_nei" if ignore_nei else "noignore_nei"}'
            config = {
                'normalize_scores': norm,
                'scale': scale, 
                'ignore_nei': ignore_nei,
                'fingerprint': fingerprint,
            }
            
            configs_judge.append(config)

if __name__ == '__main__':

    for config in configs_retrieval:
        with open(os.path.join(config['out_dir'], 'config.json'), 'w') as f:
            f.write(json.dumps(config, indent=4))

        setup_logging(config['out_dir'])
            
        ds = AuredDataset(json_data_filepath, **config)

        step_retrieval(ds=ds, config=config, golden_labels_file=golden_labels_file)
    
    logger_exp = logging.getLogger('clef.experiment')

    # find best config for retrieval
    mode = 'retrieval'
    score_by='MAP'
    for retriever in ['TERRIER', 'OPENAI']:

        configs = find_best_config_fp(f'./data-out/experiments/test-dev/retr-{retriever}', mode, score_by)
        for i, (f1, config) in enumerate(configs):
            logger_exp.info(f'{mode} {retriever} - {i+1}. {score_by} score: {f1:.4f}, config: {config}')

        best_config = configs[0][1]
        logger_exp.info(f'best config for {mode} {retriever}: {best_config}')

        for verifier_label in ['LLAMA', 'OPENAI']:
            setup_logging(best_config['out_dir'])

            best_config['verifier_label'] = verifier_label

            if 'LLAMA' in  best_config['verifier_label'].upper():
                from clef.verification.models.hf_llama3 import Llama3Verifier
                verifier = Llama3Verifier()
            elif 'OPENAI' in best_config['verifier_label'].upper():
                from clef.verification.models.open_ai import OpenaiVerifier
                verifier = OpenaiVerifier()
            
            ds = AuredDataset(json_data_filepath, **best_config)

            trec_filepath = f'{best_config["out_dir"]}/{best_config["retriever_label"]}-{best_config["split"]}.trec.txt'
            ds.add_trec_file_judgements(trec_filepath, sep=' ', normalize_scores=False) # scores are not used at this point

            verification_decisions = predict_evidence(ds, verifier)

            for c in configs_judge:
                config_jdg = {
                    **best_config,
                    **c,
                }

                ds = AuredDataset(json_data_filepath, **config_jdg)

                trec_filepath = f'{config_jdg["out_dir"]}/{config_jdg["retriever_label"]}-{config_jdg["split"]}.trec.txt'
                ds.add_trec_file_judgements(trec_filepath, sep=' ', normalize_scores=config_jdg['normalize_scores'])

                solomon = Judge(scale=config_jdg['scale'], ignore_nei=config_jdg['ignore_nei'])

                ds_grouped = ds.get_grouped_rumors()
                res_jsons = []
                for rumor_id in ds_grouped:
                    evidence_predictions = []
                    claim = ds_grouped[rumor_id]['rumor']
                    label = ds_grouped[rumor_id]['label']
                    retrieved_evidence = ds_grouped[rumor_id]['retrieved_evidence']
                    
                    for post_id, prediction in verification_decisions[rumor_id]:
                        post = None
                        for post in retrieved_evidence:
                            if post_id == post.post_id:
                                post = post
                                break
                        evidence_predictions += [(claim, post, prediction)]

                    pred_label, pred_evidence = solomon(evidence_predictions)

                    predicted = {
                        "id": rumor_id,
                        "label": label,
                        "claim": claim,
                        "predicted_label": pred_label,
                        "predicted_evidence": pred_evidence,
                    }
                    res_jsons += [predicted]
                
                if not os.path.exists(os.path.join(config_jdg['out_dir'], config_jdg['fingerprint'])):
                    os.makedirs(os.path.join(config_jdg['out_dir'], config_jdg['fingerprint'], 'eval'))

                with open(os.path.join(config_jdg['out_dir'], config_jdg['fingerprint'], 'config.json'), 'w') as f:
                    f.write(json.dumps(config_jdg, indent=4))

                verification_outfile = f'{config_jdg["out_dir"]}/{config_jdg["fingerprint"]}/zeroshot-ver-openai-retr-{config_jdg["retriever_label"]}.jsonl'
                write_jsonlines_from_dicts(verification_outfile, res_jsons)

                macro_f1, sctrict_macro_f1 = eval_run_custom(verification_outfile, json_data_filepath, '')

                from clef.pipeline.pipeline import logger
                logger.info(f'result for verification run - Strict-F1: {macro_f1:.4f} Strict-Macro-F1: {sctrict_macro_f1:.4f} with config {config_jdg} and TREC FILE {trec_filepath}')
                with open(os.path.join(config_jdg['out_dir'], config_jdg['fingerprint'], 'eval', 'log.txt'), 'a') as fh:
                    fh.write(f'result for verification run - Strict-F1: {macro_f1:.4f} Strict-Macro-F1: {sctrict_macro_f1:.4f} with config {config_jdg} and TREC FILE {trec_filepath}\n')

    # find best config for verification overall
    mode = 'verification'
    score_by='F1'
    configs = find_best_config_fp(f'./data-out/experiments/test-dev', mode, score_by)
    for i, (f1, config) in enumerate(configs):
        logger_exp.info(f'{mode} - {i+1}. {score_by} score: {f1:.4f}, config: {config}')
    
    try:
        no_ext_data = []
        ext_data = []
        for item in configs:
            config = item[1]
            if config['add_author_name'] or config['add_author_bio']:
                ext_data.append(item)
            else:
                no_ext_data.append(item)
        
        no_ext_data = sorted(no_ext_data, key=lambda x: x[0], reverse=True)
        ext_data = sorted(ext_data, key=lambda x: x[0], reverse=True)

        for i, (f1, config) in enumerate(no_ext_data):
            logger_exp.info(f'{mode} no ext data - {i+1}. {score_by} score: {f1:.4f}, config: {config["out_dir"]}/{config["fingerprint"]} - {config}')

        for i, (f1, config) in enumerate(ext_data):
            logger_exp.info(f'{mode} using ext data - {i+1}. {score_by} score: {f1:.4f}, config: {config["out_dir"]}/{config["fingerprint"]} - {config}')    

    except:
        pass