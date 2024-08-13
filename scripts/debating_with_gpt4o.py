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
        self.conversation = [{"role": "system", "content": """You are a misinformation detection expert in the news domain. You will look at image-caption pairs and decide if the given image is rightly used in the given news context. To further assist you, a summary of news articles related to the image will be provided.
                    Based on this, you need to decide if the caption belongs to the image or if it is being used to spread false information to mislead people.
                    Note that the image is real. It has not been digitally altered. 
                    Carefully examine the image for any known entities, people, watermarks, dates, landmarks, flags, text, logos and other details which could give you important information to better explain your answer.
                    Remember in news articles images and captions are often related contextually and the caption need not exactly describe the image.
                    The goal is to consider the contextual relationship between the image and caption based on the news articles and correctly identify if the image caption pair is misinformation or not and to explain your answer in detail.
                    Think step by step and plan a detailed explanation for your answer."""}]
        self.model = model_name
        self.client = OpenAI(api_key="sk-wT0iyZwk5iii4Uo3jM0d894RQ5LKyTBNwii6-IMdkgT3BlbkFJswQud0sa3DgTshimuy0yTL46fTAdoD3tfI3pj5BEEA")
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
    model_b = OpenAIModel(model_name)
    for data_idx in tqdm(range(args.start_idx, args.end_idx)):
        image, caption, image_path, annotation = get_data(data_idx)
        summary_key = str(annotation['id'])+"_"+str(annotation["image_id"])
        summary = retrieve_summary(summary_key)
        encoded_image_path = encode_image(image_path)
        prompts = ["""This is the caption: {}. This is the summary of true news articles related to the image: {}.
                    Now based on all the information, give a definite YES or NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?
                    """.format(caption, summary),
                    """This is what I think about the same image-caption pair: [RESP].
                                        Do you agree with me? If you think I am wrong then convince me why.
                                        Clearly state your reasoning and tell me if I am missing out on some important information or am making some logical error.
                                        Use this information to improve/correct your answer.
                                        At the end give a definite YES or NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?""",
                    """I see what you mean and this is what I think: [RESP]. Do you agree with me?
                                    If not then point out the inconsistencies in my argument (e.g. location, time or person related logical confusion) and explain why you are correct. 
                                    If you disagree with me then clearly state why and what information I am overlooking.
                                    I want you to help me improve my argument and explanation. 
                                    Don't give up your original opinion without clear reasons, DO NOT simply agree with me without proper reasoning.
                                    At the end give a definite YES or NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?""",
                                    """I see what you mean and this is what I think: [RESP]. Do you agree with me?
                                    If not then point out the inconsistencies in my argument (e.g. location, time or person related logical confusion) and explain why you are correct. 
                                    If you disagree with me then clearly state why and what information I am overlooking.
                                    I want you to help me improve my argument and explanation. 
                                    Don't give up your original opinion without clear reasons, DO NOT simply agree with me without proper reasoning.
                                    At the end give a definite YES or NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?""",
                                    """I see what you mean and this is what I think: [RESP]. Do you agree with me?
                                    If not then point out the inconsistencies in my argument (e.g. location, time or person related logical confusion) and explain why you are correct. 
                                    If you disagree with me then clearly state why and what information I am overlooking.
                                    I want you to help me improve my argument and explanation. 
                                    Don't give up your original opinion without clear reasons, DO NOT simply agree with me without proper reasoning.
                                    At the end give a definite YES or NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?"""
                    ]
        model_a.add_message("user", [{"type":"text", "text":"This is the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_path}"}}])
        model_b.add_message("user", [{"type":"text", "text":"This is the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_path}"}}])
        
        model_responses = {}
        error = False
        for i in range(args.num_models):
            model_responses[i] = {"falsified":"", "output":""}
        for round in range(args.num_rounds+1):
            if round == 0:
                output_a = model_a.generate_response(prompts[round])
                    #print("AGENT-1: ", output_a)
                output_b = model_b.generate_response(prompts[round])
                    #print("AGENT-2: ", output_b)
            else:
                prompt_a = prompts[round].replace("[RESP]", output_b)
                prompt_b = prompts[round].replace("[RESP]", output_a)
                output_a = model_a.generate_response(prompt_a)
                #print("AGENT-1: ", output_a)
                output_b = model_b.generate_response(prompt_b)
                #print("AGENT-2: ", output_b)
            if "YES" in output_a or "Yes" in output_a:
                model_responses[0]["falsified"] = True
                model_responses[0]["output"] = output_a
            elif "NO" in output_a or "No" in output_a:
                model_responses[0]["falsified"] = False
                model_responses[0]["output"] = output_a
            if "YES" in output_b or "Yes" in output_b:
                model_responses[1]["falsified"] = True
                model_responses[1]["output"] = output_b
            elif "NO" in output_b or "No" in output_b:
                model_responses[1]["falsified"] = False
                model_responses[1]["output"] = output_b
            if ("YES" in output_a or "Yes" in output_a) and ("YES" in output_b or "Yes" in output_b):
                #print("********** Models agree on YES ************")
                break
            elif ("NO" in output_a or "No" in output_a) and ("NO" in output_b or "No" in output_b):
                #print("********** Models agree on NO ************")
                break
        if error:
            annotation['falsified'] = "ERROR"
            annotation['output'] = "NA"
        else:
            annotation['falsified'], annotation["output"] = get_final_prediction(args.num_models, model_responses)
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
                
