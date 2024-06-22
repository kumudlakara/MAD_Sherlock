import sys
sys.path.append("..")

import http.client, urllib.parse
import requests
import json
import os.path
import argparse
from tqdm import tqdm

import torch
import transformers
from transformers import AutoTokenizer
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate

from utils.data import get_data
from utils.external_retrieval import get_matching_urls, get_webpage_text

BASE_URL = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
SUBSCRIPTION_KEY = "fcbb57be03f64cf191aeb1ba7835fc04"

def store_data(key, value, file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except:
        data = {}
    
    data[key] = value

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def entry_exists(key, file_path="./retrieval_urls.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
        if key in list(data.keys()):
            if data[key] != []:
                return True
            else:
                return False   
        else:
            return False     

def main(args):
    for i in tqdm(range(args.start_idx, args.end_idx)):
        image, caption, image_path, annotation = get_data(i)
        file = {'image' : ('myfile', open(image_path, 'rb'))}
        key = str(annotation['id'])+'_'+str(annotation['image_id'])
        if not entry_exists(key):
            matching_urls = get_matching_urls(file, BASE_URL, SUBSCRIPTION_KEY)
            store_data(key, matching_urls, args.data_file)
        else:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--data_file", type=str, default="./retrieval_urls.json")
    args = parser.parse_args()
    main(args)

    



