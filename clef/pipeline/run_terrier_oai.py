import os
from clef.utils.logging_setup import setup_logging 


from clef.pipeline.pipeline import step_retrieval, step_verification
from clef.utils.data_loading import AuredDataset

#
# define parameters
#

root_path = '../../' # path to github repository root level (where setup.py is located)

config = {
    "blind_run": True,
    "split": "test",
    "add_author_name": False,
    "add_author_bio": False,
    "retriever_k": 5,
    "out_dir": "./data-out/runs-test/terrier-oai",
    "preprocess": True,
    "retriever_label": "TERRIER",
    "verifier_label": "OPENAI",
    "normalize_scores": False,
    "scale": False,
    "ignore_nei": False,
    "fingerprint": "nonorm-noscale-ignore_nei"
}

# ensure out_dir directories exist for saving output (required for anserini, etc - not only for eval)
if not os.path.exists(config['out_dir']):
    os.makedirs(config['out_dir'])
    if not os.path.exists(os.path.join(config['out_dir'], 'eval')):
        os.makedirs(os.path.join(config['out_dir'], 'eval'))

setup_logging(config['out_dir'])

data_path = os.path.join(root_path, 'clef2024-checkthat-lab', 'task5', 'data')
json_data_filepath = os.path.join(data_path, 'English_test.json')
    
ds = AuredDataset(json_data_filepath, **config)

step_retrieval(ds=ds, config=config, golden_labels_file=None)

step_verification(ds=ds, config=config, ground_truth_filepath=None)