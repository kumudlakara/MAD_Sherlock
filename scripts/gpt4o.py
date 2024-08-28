import sys
sys.path.append('..')

import argparse
import torch

import requests
from PIL import Image
import json
import base64
from io import BytesIO
from tqdm import tqdm
import time

from openai import OpenAI

from utils.data import get_data, show_data
from utils.stored_retrieval import retrieve_summary
from utils.compile_results import add_result

def get_final_prediction(num_models, model_responses):
    num_true, num_false, num_unsure = 0,0,0
    final_pred = ""
    final_outputs = {}
    for i in range(num_models):
        if model_responses[i]["falsified"] == True:
            num_true += 1
        elif model_responses[i]["falsified"] == False:
            num_false += 1
        else:
            num_unsure += 1
        final_outputs["model_"+str(i)] = model_responses[i]["output"]

    if num_true > num_false and num_true > num_unsure:
        final_pred = True
    elif num_false > num_true and num_false > num_unsure:
        final_pred = False
    else:
        final_pred = "Unsure"
    return final_pred, final_outputs

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class OpenAIModel:
    def __init__(self,model_name):
        # Initialize conversation with a system message
        self.conversation = [{"role": "system", "content": """You are a misinformation detection expert in the news domain. 
                    You need to decide if a given caption belongs to a given image or if it is being used to spread false information to mislead people."""}]
        self.model = model_name
        self.client = OpenAI(api_key="***")
    def add_message(self, role, content):
        # Adds a message to the conversation.
        self.conversation.append({"role": role, "content": content})
    def generate_response(self, prompt):
        # Add user prompt to conversation
        self.add_message("user", prompt)
        # Make a request to the API using the chat-based endpoint with conversation context
        response = self.client.chat.completions.create(model=self.model, messages=self.conversation, max_tokens=512)
        # Extract the response
        assistant_response = response.choices[0].message.content
        # Add assistant response to conversation
        self.add_message("assistant", assistant_response)
        # Return the response
        return assistant_response

def main(args):
    model_name = "gpt-4o"
    model_a = OpenAIModel(model_name)
    j = args.start_idx
    while j < args.end_idx:
        print(j)
        data_idx = j
        image, caption, image_path, annotation = get_data(data_idx)
        encoded_image_path = encode_image(image_path)
        prompts = ["""This is the caption: {}.
                    Now based on all the information, give a definite YES or NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?
                    """.format(caption)]
        model_a.add_message("user", [{"type":"text", "text":"This is the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_path}"}}])
        try:
            output_a = model_a.generate_response(prompts[0])
            j += 1
        except:
            print("sleeping now!")
            time.sleep(180)
            model_a = OpenAIModel(model_name)
        if "YES" in output_a or "Yes" in output_a:
            annotation['falsified'] = True
        elif "NO" in output_a or "No" in output_a:
            annotation['falsified'] = False
        annotation["output"] = output_a
        
        add_result(args.result_file, annotation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--num_models", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--result_file", type=str, default="results.json")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    args = parser.parse_args()
    main(args)