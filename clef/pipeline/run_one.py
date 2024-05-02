import os
from clef.utils.logging_setup import setup_logging 


from clef.pipeline.pipeline import step_retrieval, step_verification
from clef.utils.data_loading import AuredDataset

#
# define parameters
#

root_path = '../../' # path to github repository root level (where setup.py is located)

config = {
    'blind_run': True,
    'split': 'test',
    'preprocess': True,
    'add_author_name': False,
    'add_author_bio': False,
    'out_dir': './data-out/runs/zero',
    'retriever_k': 5,
    'retriever_label': 'OPENAI',
    'normalize_scores': True,
    'scale': False, 
    'ignore_nei': True,
}

setup_logging(config['out_dir'])


data_path = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data')
json_data_filepath = os.path.join(data_path, 'English_test.json')
    
ds = AuredDataset(json_data_filepath, **config)
# ds.rumors = ds.rumors[0:2] # subset here

step_retrieval(ds=ds, config=config, golden_labels_file=None)

step_verification(ds=ds, config=config, ground_truth_filepath=None)