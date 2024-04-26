import os
from clef.pipeline.pipeline import run_pipe, ver

import logging
# Setting up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./pipeline.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)


#
# define parameters
#
root_path = '../../' # path to github repository root level (where setup.py is located)

task5_dir = 'clef2024-checkthat-lab/task5'

data_path = os.path.join(root_path, task5_dir, 'data') # relative to root

# filepath = os.path.join(data_path, 'English_train.json') # relative to root
filepath = os.path.join(data_path, 'English_dev.json') # relative to root

golden_labels_file = os.path.join(root_path, task5_dir, 'data', 'dev_qrels.txt') # relative to root

# for preprocess in [True, False]:
#     for add_author_name in [True, False]:
#         for add_author_bio in [True, False]:
#             for retriever_type in ['TFIDF', 'SBERT', 'OPENAI']: #,'LUCENE']
#                 config = {
#                     'preprocess': preprocess,
#                     'add_author_name': add_author_name,
#                     'add_author_bio': add_author_bio,
#                     'out_dir': './data-out/setup1-exp',
#                     'retriever_k': 5,
#                     'retriever_label': retriever_type
#                 }
#                 run_pipe(filepath, golden_labels_file, config)

for preprocess in [True]:
    for add_author_name in [True]:
        for add_author_bio in [True]:
            # for retriever_type in ['OPENAI', 'TFIDF', 'SBERT', 'LUCENE']:
            for retriever_type in ['OPENAI']: 
                config = {
                    'preprocess': preprocess,
                    'add_author_name': add_author_name,
                    'add_author_bio': add_author_bio,
                    'out_dir': './data-out/setup1-exp',
                    'retriever_k': 5,
                    'retriever_label': retriever_type,
                    'normalize_scores': True,
                    'scale': False, 
                    'ignore_nei': True,
                }
                # run_pipe(filepath, golden_labels_file, config)
                ver(filepath, config)