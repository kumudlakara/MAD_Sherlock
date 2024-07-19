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
from peft import PeftModel

from PIL import Image
from tqdm import tqdm
import argparse
import json

from utils.data import get_data, show_data
from utils.external_retrieval import get_matching_urls, get_summary, get_query_answer, get_webpage_text
from utils.compile_results import add_result
from utils.prompts import initial_prompt_actor, initial_prompt_skeptic, refine_actor_response_prompt, end_decision_prompt
from utils.stored_retrieval import retrieve_summary, retrieve_stored_url

def get_conv_and_roles(model_name, conv_mode):
    conv = []
    roles = []

    for i in range(2):
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

def main(args):
    models = []
    
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    for i in range(2):
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base=None, model_name=model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit, device_map="auto",
                                                                               max_memory={0:"20000MiB",1:"20000MiB",2:"20000MiB",3:"20000MiB",4:"20000MiB",
                                                                                           5:"20000MiB",6:"20000MiB",7:"35000MiB"})
        if args.load_finetuned:
            model = PeftModel.from_pretrained(model, args.finetuned_model_path, device_map="auto")
            model.merge_and_unload()
            model.to(dtype=torch.bfloat16)
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
        num_passes = 0
        conv, roles = get_conv_and_roles(model_name, conv_mode)
        image, caption, img_path, annotation = get_data(data_idx)
        summary_key = str(annotation['id'])+"_"+str(annotation["image_id"])
        context = retrieve_summary(summary_key)
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], models[0]['image_processor'], models[0]['model'].config)
        if type(image_tensor) is list:
            image_tensor = [image.to(models[0]['model'].device, dtype=torch.bfloat16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(models[0]['model'].device, dtype=torch.bfloat16)

        actor_inp = initial_prompt_actor(roles[0][0], caption, context)
        if image is not None:
            # first message
            if models[0]['model'].config.mm_use_im_start_end:
                actor_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + actor_inp
            else:
                actor_inp = DEFAULT_IMAGE_TOKEN + '\n' + actor_inp

        conv[0].append_message(conv[0].roles[0], actor_inp)
        conv[0].append_message(conv[0].roles[1], None)
        actor_outputs = generate_output(0, conv, models, image_tensor, args.temperature, image_size, args.max_new_tokens)
        conv[0].messages[-1][-1] = actor_outputs
        prev_response = True
        if "NO" in actor_outputs or "No" in actor_outputs:
            prev_response = False
        prev_ques = ""
        while num_passes != 3:
            skeptic_inp = initial_prompt_skeptic(roles[1][0], actor_outputs)
            if image is not None:
                # first message
                if models[1]['model'].config.mm_use_im_start_end:
                    skeptic_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + skeptic_inp
                else:
                    skeptic_inp = DEFAULT_IMAGE_TOKEN + '\n' + skeptic_inp
                image = None
            conv[1].append_message(conv[1].roles[0], skeptic_inp)
            conv[1].append_message(conv[1].roles[1], None)
            skeptic_outputs = generate_output(1, conv, models, image_tensor, args.temperature, image_size, args.max_new_tokens)
            if skeptic_outputs == prev_ques:
                break
            else:
                prev_ques = skeptic_outputs
            conv[1].messages[-1][-1] = skeptic_outputs

            actor_inp = refine_actor_response_prompt(roles[0][0], skeptic_outputs)
            if image is not None:
                # first message
                if models[0]['model'].config.mm_use_im_start_end:
                    actor_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + actor_inp
                else:
                    actor_inp = DEFAULT_IMAGE_TOKEN + '\n' + actor_inp

            conv[0].append_message(conv[0].roles[0], actor_inp)
            conv[0].append_message(conv[0].roles[1], None)
            actor_outputs = generate_output(0, conv, models, image_tensor, args.temperature, image_size, args.max_new_tokens)
            conv[0].messages[-1][-1] = actor_outputs
            skeptic_inp = end_decision_prompt(roles[1][0], prev_response, actor_outputs)
            if image is not None:
                # first message
                if models[1]['model'].config.mm_use_im_start_end:
                    skeptic_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + skeptic_inp
                else:
                    skeptic_inp = DEFAULT_IMAGE_TOKEN + '\n' + skeptic_inp
            conv[1].append_message(conv[1].roles[0], skeptic_inp)
            conv[1].append_message(conv[1].roles[1], None)
            skeptic_outputs = generate_output(1, conv, models, image_tensor, args.temperature, image_size, args.max_new_tokens)
            conv[1].messages[-1][-1] = skeptic_outputs
            if 'PERFECT' in skeptic_outputs:
                break
            else:
                num_passes += 1
                if "NO" in actor_outputs or "No" in actor_outputs:
                    prev_response = False
                else:
                    prev_response = True
                continue
        if "NO" in actor_outputs or "No" in actor_outputs:
            annotation['falsified'] = False
        elif "YES" in actor_outputs or "Yes" in actor_outputs:
            annotation['falsified'] = True
        else:
            annotation['falsified'] = "Unsure"
        annotation['output'] = actor_outputs
        add_result(args.save_file, annotation)
                   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--load_finetuned", type=bool, default=True)
    parser.add_argument("--finetuned_model_path", type=str, default="../../datasets/models/checkpoints/llava-v1_6_34b_finetuning_2/checkpoint-6000/")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--save_file", type=str, default="results.json")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    args = parser.parse_args()
    main(args)
