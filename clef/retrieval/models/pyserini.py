from pyserini.search.lucene import LuceneSearcher
import os
import json
import shutil
import subprocess

def searchPyserini(rumor_id,
                   query,
                   timeline,
                   k = 5,
                   temp_dir_path = 'temp',
                   index_path = 'temp/index',
                   cleanup_temp_dir = True):
    
    # if you get the error "NameError: name '_C' is not defined" --> restart the Jupyter Kernel
    # if you get the error "...Java VM is already running..."    --> restart the Jupyter Kernel
    
    # ensure "working directory" exists (where we store intermediate data like the dynamic index that will be quered later)
    if not os.path.exists(temp_dir_path):
        os.mkdir(temp_dir_path)

    # set up "dynamic" (= temporary) index file using timeline data
    dynamic_index_filename = 'dynamic-index.jsonl'
    with open(os.path.join(temp_dir_path, dynamic_index_filename), mode='w', encoding='utf8') as file:
        for tweet in timeline:
            id = tweet[1]
            text = tweet[2]
            file.write(json.dumps({'id': id, 'contents': text}) + '\n')
    
    # ensure index directory exists and is empty
    if os.path.exists(index_path):
        for filename in os.listdir(index_path):
            if os.path.isfile(os.path.join(index_path, filename)):
                os.remove(os.path.join(index_path, filename))
    else:
        os.mkdir(index_path)

    # set up pyserini command since python embeddable is not released yet
    nthreads = 1
    command = f'python -m pyserini.index.lucene ' \
    f'-input {temp_dir_path} ' \
    f'-collection JsonCollection ' \
    f'-generator DefaultLuceneDocumentGenerator ' \
    f'-index {index_path} ' \
    f'-threads {nthreads} ' \
    f'-storePositions ' \
    f'-storeDocvectors ' \
    f'-storeRaw ' \
    f'-language en'

    result = subprocess.run(command, capture_output=True)

    # intialize searcher using index directoy
    searcher = LuceneSearcher(index_path)
    hits = searcher.search(query)

    ranked = []

    for i, hit in enumerate(hits[:k]):
        ranked += [[rumor_id, hit.docid, i+1, hit.score]]

        # print debugging data
        # doc = searcher.doc(hit.docid)
        # json_doc = json.loads(doc.raw())
        # wrap(f'{i+1:2} {hit.docid:4} {hit.score:.5f}\n{json_doc["contents"]}')

    if cleanup_temp_dir:
        shutil.rmtree('temp/')

    return ranked