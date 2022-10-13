from typing import Counter
from elasticsearch import Elasticsearch
import json
import urllib.request
import os
import argparse
from timeit import default_timer as timer
import shutil
from functools import cache
import time
from tqdm import tqdm
from vars import url, api_key_1, api_key_2
import logging
from torchreid.utils import FeatureExtractor
from datetime import datetime
import numpy
'''
Takes in a query and adds the feature vectors into elastic search
query can be dynamically ajusted based in time frame. Currently feature vectors are only
used on
'''


name = 'image'

input_path = f"./media/{name}/"

es = Elasticsearch(url, api_key=(api_key_1, api_key_2))

f = open('query.json',)
search_query = json.load(f)
global_end_time = datetime.now().isoformat()
global_start_time = '2022-10-13T07:17:15.892850'


def main():
    start_time = time.time()
    global_end_time = datetime.now().isoformat()
    #search_query['query']['bool']['filter'][1]['range']['inferenced_timestamp']['lte'] = global_end_time
    json_info = es.search(index = "snl-ghostrunner-*", body = search_query)
    elastic_docs = json_info["hits"]["hits"]
    # iterate the docs returned by API call
    instances = {}
    classifications = set()
    if os.path.isdir(f'{input_path}') == False:
        os.makedirs(f'{input_path}')
    print("Images Are Downloading")
    counter = 0
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(elastic_docs, f, ensure_ascii=False, indent=4)
    for num, doc in enumerate(tqdm(elastic_docs)):
        url_of_image = str(doc['fields']['s3_presigned'][0])
        instances_id = doc['_id']
        index = doc['_index']
        full_file_name = os.path.join(input_path, f"{instances_id}.jpg")
        urllib.request.urlretrieve(url_of_image, full_file_name)
        image = full_file_name
        extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='./osnet_ms_d_c.pth.tar',
        device='cuda'
        )        
        features = extractor(image) 
        features = features.cpu().detach().numpy()
        features = features[0]
        #features = numpy.set_printoptions(suppress=True)
        document = {'person_vectors': features}
        #counter +=1
        # print(type(features))
        # print(features.shape)
        #print(features)
        # print(index)
        #counter += 1
        try:
            es.update(id = instances_id,
            index = index,
            doc = document,
            request_timeout= 60
            )
        except Exception as e: 
            logging.critical(e)            
            logging.warning('Failed to index') 

if __name__ == '__main__':
    start = timer()
    main()
    dir = './media/'
    shutil.rmtree(dir)
    end = timer() 
    print(f"Process finished --- {start - end} seconds ---")
    print(f"time per per request  {(start - end)}")