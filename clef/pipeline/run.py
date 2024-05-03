import os
from clef.utils.logging_setup import setup_logging 


from clef.pipeline.pipeline import step_retrieval, step_verification
from clef.utils.data_loading import AuredDataset

#
# define parameters
#

root_path = '../../' # path to github repository root level (where setup.py is located)

config = {
    'blind_run': False,
    'split': 'dev',
    'preprocess': True,
    'add_author_name': False,
    'add_author_bio': False,
    'out_dir': './data-out/runs/dev',
    'retriever_k': 5,
    'retriever_label': 'OPENAI',
    'verifier_label': 'OPENAI',
    'normalize_scores': True,
    'scale': False, 
    'ignore_nei': True,
}

# ensure out_dir directories exist for saving output (required for anserini, etc - not only for eval)
if not os.path.exists(config['out_dir']):
    os.makedirs(config['out_dir'])
    if not os.path.exists(os.path.join(config['out_dir'], 'eval')):
        os.makedirs(os.path.join(config['out_dir'], 'eval'))

setup_logging(config['out_dir'])


data_path = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data')
json_data_filepath = os.path.join(data_path, 'English_dev.json')
golden_path =  os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data', 'dev_qrels.txt')
    
ds = AuredDataset(json_data_filepath, **config)
# ds.rumors = ds.rumors[0:2] # subset here

step_retrieval(ds=ds, config=config, golden_labels_file=golden_path)

step_verification(ds=ds, config=config, ground_truth_filepath=json_data_filepath)