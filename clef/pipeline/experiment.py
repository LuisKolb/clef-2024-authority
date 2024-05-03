
import os
from clef.utils.logging_setup import setup_logging 


from clef.pipeline.pipeline import step_retrieval, step_verification
from clef.utils.data_loading import AuredDataset

#
# define parameters
#

root_path = '../../' # path to github repository root level (where setup.py is located)

json_data_filepath = os.path.join(root_path, 'clef', 'data', 'combined_data.jsonl')

golden_labels_file = os.path.join(root_path, 'clef', 'data', 'combined_qrels.txt') # relative to root

for preprocess in [True, False]:
    for norm in [True, False]:
        for scale in [True, False]:
            for ignore_nei in [True, False]:
                config = {
                    'blind_run': False,
                    'split': 'combined',
                    'preprocess': preprocess,
                    'add_author_name': False,
                    'add_author_bio': False,
                    'out_dir': f'./data-out/experiments/PreNormScaleNei/{1 if preprocess else 0}{1 if norm else 0}{1 if scale else 0}{1 if ignore_nei else 0}',
                    'retriever_k': 5,
                    'retriever_label': 'TERRIER',
                    'verifier_label': 'LLAMA',
                    'normalize_scores': norm,
                    'scale': scale, 
                    'ignore_nei': ignore_nei,
                }

                # ensure out_dir directories exist for saving output (required for anserini, etc - not only for eval)
                if not os.path.exists(config['out_dir']):
                    os.makedirs(config['out_dir'])
                    if not os.path.exists(os.path.join(config['out_dir'], 'eval')):
                        os.makedirs(os.path.join(config['out_dir'], 'eval'))

                setup_logging(config['out_dir'])
                    
                ds = AuredDataset(json_data_filepath, **config)

                r5, meanap = step_retrieval(ds=ds, config=config, golden_labels_file=golden_labels_file)

                macro_f1, sctrict_macro_f1 = step_verification(ds=ds, config=config, ground_truth_filepath=json_data_filepath)