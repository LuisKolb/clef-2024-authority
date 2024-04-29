import clef.utils.logging_setup # set up logging

import os
from clef.pipeline.pipeline import step_retrieval, step_verification
from clef.utils.data_loading import AuredDataset

#
# define parameters
#

root_path = '../../' # path to github repository root level (where setup.py is located)

data_path = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data')

config = {
    'blind_run': True,
    'split': 'test',
    'preprocess': False,
    'add_author_name': False,
    'add_author_bio': False,
    'out_dir': './data-out/runs/one',
    'retriever_k': 5,
    'retriever_label': 'OPENAI',
    'normalize_scores': True,
    'scale': False, 
    'ignore_nei': True,
}

json_data_filepath = os.path.join(data_path, 'English_test.json') # relative to root
    
ds = AuredDataset(json_data_filepath, **config)

step_retrieval(ds=ds, config=config, golden_labels_file=None)

step_verification(ds=ds, config=config, ground_truth_filepath=None)