"""
Models always have web context to begin with.
"""

import sys
sys.path.append('..')

import torch
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import torch
import transformers
from transformers import AutoTokenizer, pipeline
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate

from PIL import Image
from tqdm import tqdm
import argparse
import json

from utils.data import get_data, show_data
from utils.external_retrieval import get_matching_urls, get_summary, get_query_answer, get_webpage_text
from utils.compile_results import add_result
from utils.prompts import round1_prompt_with_disambiguation, debate_prompt_with_disambiguation, initial_prompt_with_context, refine_prompt
from utils.stored_retrieval import retrieve_summary, retrieve_stored_url

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

def get_query_answer(matching_urls, query):
    prompt_template = """
            Based on the text delimited by triple backticks, answer this question: {query}
            ```{text}```
            ANSWER:
            """
    prompt = PromptTemplate(template=prompt_template, input_variables=['query', 'text'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    texts = []
    for matching_url in matching_urls:
        try:
            texts.append(get_webpage_text(matching_url))
        except:
            continue
    text = ""
    print("Found {} search results.".format(len(texts)))
    if len(texts) == 0:
        #to account for cases where no search results are found
        return "No search results found, cannot answer query."
    for i in range(len(texts)):
        #only take top k=3 articles
        if i == 3: 
            break
        if "the" not in texts[i]:
            #naive way to ensure text is in English
            continue
        text += "\n\n"
        text += texts[i]
    output = llm_chain.run({'query':query, 'text': text})
    return output[output.find("ANSWER"):].rstrip()


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

def get_conv_and_roles(model_name, conv_mode):
    conv = []
    roles = []

    for i in range(args.num_models):
        conv.append(conv_templates[conv_mode].copy())
        if "mpt" in model_name.lower():
            roles.append(('user', 'assistant'))
        else:
            roles.append(conv[i].roles)
    return conv, roles


def generate_output(i, conv, models, image_tensor, temperature, image_size, max_new_tokens):
    prompt = conv[i].get_prompt()

    input_ids = tokenizer_image_token(prompt, models[i]['tokenizer'], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(models[i]['model'].device)
    stop_str = conv[i].sep if conv[i].sep_style != SeparatorStyle.TWO else conv[i].sep2
    keywords = [stop_str]

    with torch.inference_mode():
        output_ids = models[i]['model'].generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True)

    outputs = models[i]['tokenizer'].decode(output_ids[0]).strip()
    return outputs

def refine_response(data_annotation, i, conv, models, image_tensor, temperature, image_size, max_new_tokens, role, query, prev_response):
    key = str(data_annotation['id'])+"_"+str(data_annotation['image_id'])
    matching_urls = retrieve_stored_url(key)
    search_result = get_query_answer(matching_urls, query)
    if "disambiguation" in prev_response:
        prev_response = prev_response[:prev_response.find("disambiguation")]
    elif "Disambiguation" in prev_response:
        prev_response = prev_response[:prev_response.find("Disambiguation")]
    else:
        prev_response = prev_response[:prev_response.find("<search_query>")]
    if "no results found" in search_result:
        return conv, prev_response, search_result
    inp = refine_prompt(role, query, search_result, prev_response)
    conv[i].append_message(conv[i].roles[0], inp)
    conv[i].append_message(conv[i].roles[1], None)
    outputs = generate_output(i, conv, models, image_tensor, temperature, image_size, max_new_tokens)
    conv[i].messages.pop()
    conv[i].messages.pop()
    return conv, outputs, search_result

def main(args):
    models = []
    
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    for i in range(args.num_models):
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base=None, model_name=model_name, 
                                    load_8bit=args.load_8bit, load_4bit=args.load_4bit, device_map="auto")
        models.append({"tokenizer":tokenizer, "model":model, "image_processor":image_processor, "context_len":context_len})

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    print("Running inference now!")
    for data_idx in tqdm(range(args.start_idx, args.end_idx)):
        search_result = ""
        search_done = False
        conv, roles = get_conv_and_roles(model_name, conv_mode)
        image, caption, img_path, annotation = get_data(data_idx)
        summary_key = str(annotation['id'])+"_"+str(annotation["image_id"])
        context = retrieve_summary(summary_key)
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], models[0]['image_processor'], models[0]['model'].config)
        if type(image_tensor) is list:
            image_tensor = [image.to(models[0]['model'].device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(models[0]['model'].device, dtype=torch.float16)
        
        temp = ""
        model_responses = {}
        for i in range(args.num_models):
            model_responses[i] = {"falsified":"", "output":""}
            queries = {0:"", 1:""}
            search_results = {0: "", 1: ""}
        for round in range(args.num_rounds+1):
            queries = {0:"", 1:""}
            for i in range(args.num_models):
                if round == 0:
                    inp = initial_prompt_with_context(roles[i][0], caption, context)   
                elif round == 1:
                    if i == 1:
                        inp = round1_prompt_with_disambiguation(roles[i][0], temp)
                    else:
                        inp = round1_prompt_with_disambiguation(roles[i][0], conv[(i+1)%args.num_models].messages[-1][-1])
                else:
                    inp = debate_prompt_with_disambiguation(roles[i][0], conv[(i+1)%args.num_models].messages[-1][-1])
                if image is not None:
                    # first message
                    if models[i]['model'].config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                    if round == 0 and i == 1:
                        image = None

                conv[i].append_message(conv[i].roles[0], inp)
                conv[i].append_message(conv[i].roles[1], None)
                outputs = generate_output(i, conv, models, image_tensor, args.temperature, image_size, args.max_new_tokens)
                conv[i].messages[-1][-1] = outputs
                
                if "<search_query>" in outputs:
                    queries[(i+1)%args.num_models] = outputs[outputs.find("<search_query>")+len("<search_query>"):outputs.find("</search_query>")]
                
                if i == 0 and round == 0:
                    temp = outputs
                
                #final answer from the model
                if "YES" in outputs or "Yes" in outputs:
                    model_responses[i]["falsified"] = True
                    model_responses[i]["output"] = outputs
                else:
                    model_responses[i]["falsified"] = False
                    model_responses[i]["output"] = outputs
            
            #disambiguate and refine responses
            if queries[0] != "":
                conv, outputs, search_res = refine_response(annotation, 0, conv, models, image_tensor, args.temperature, image_size, args.max_new_tokens, roles[0][0], queries[0], conv[0].messages[-1][-1])
                if "no results found" not in search_res:
                    model_responses[0]['outputs'] = outputs
                    conv[0].messages[-1][-1] = outputs
                search_results[0] = search_res
                if "YES" in outputs or "Yes" in outputs:
                    model_responses[0]["falsified"] = True
                    model_responses[0]["output"] = outputs
                else:
                    model_responses[0]["falsified"] = False
                    model_responses[0]["output"] = outputs
            if queries[1] != "":
                conv, outputs, search_res = refine_response(annotation, 1, conv, models, image_tensor, args.temperature, image_size, args.max_new_tokens, roles[1][0], queries[1], conv[1].messages[-1][-1])
                if "no results found" not in search_res:
                    model_responses[1]['outputs'] = outputs
                    conv[1].messages[-1][-1] = outputs
                search_results[1] = search_res
                if "YES" in outputs or "Yes" in outputs:
                    model_responses[1]["falsified"] = True
                    model_responses[1]["output"] = outputs
                else:
                    model_responses[1]["falsified"] = False
                    model_responses[1]["output"] = outputs
            
            if model_responses[0]['falsified'] == model_responses[1]['falsified'] and round != 0:
                #print("******************* Models agree!! ******************")
                break
        annotation["search_queries"] = queries
        annotation["search_results"] = search_results
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
