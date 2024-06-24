import json

def retrieve_stored_url(key, file_path="../utils/retrieval_urls.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[key]

def retrieve_summary(key, file_path="../utils/summaries.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[key]