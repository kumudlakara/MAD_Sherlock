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
SUBSCRIPTION_KEY = "***"

MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipeline = transformers.pipeline("text-generation",
                    model=MODEL_NAME,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    truncation=True,
                    max_length=3000,
                    max_new_tokens=1000,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id)
llm = HuggingFacePipeline(pipeline=pipeline)
prompt_template = """
            Write a summary of the following text delimited by triple backticks.
            Your response should cover all keypoints.
            ```{text}```
            SUMMARY:
            """
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def store_data(key, value, file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except:
        data = {}
    
    data[key] = value

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_summary(matching_urls):
    texts = []
    for matching_url in matching_urls:
        texts.append(get_webpage_text(matching_url))
    text = ""
    print("Found {} search results.".format(len(texts)))
    if len(texts) == 0:
        #to account for cases where no search results are found
        return "No search results found"
    for i in range(len(texts)):
        #only take top k=3 articles
        if i == 3: 
            break
        if "the" not in texts[i]:
            #naive way to ensure text is in English
            continue
        text += "\n\n"
        text += texts[i]
    output = llm_chain.run(text)
    return output[output.find("SUMMARY"):].rstrip()

def main(args):
    for i in tqdm(range(args.start_idx, args.end_idx)):
        image, caption, image_path, annotation = get_data(i)
        file = {'image' : ('myfile', open(image_path, 'rb'))}
        matching_urls = get_matching_urls(file, BASE_URL, SUBSCRIPTION_KEY)
        #print(matching_urls)
        search_results = get_summary(matching_urls)
        key = str(annotation['id'])+'_'+str(annotation['image_id'])
        store_data(key, search_results, args.data_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=1000)
    parser.add_argument("--end_idx", type=int, default=7000)
    parser.add_argument("--data_file", type=str, default="./summaries.json")
    args = parser.parse_args()
    main(args)

    



