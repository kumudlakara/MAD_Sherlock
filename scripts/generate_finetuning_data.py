import sys
sys.path.append('..')

import numpy as np
import json
from PIL import Image
from tqdm import tqdm

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from utils.data import get_data_elements
from utils.prompts import finetuning_datagen_prompt
from utils.compile_results import add_result

def get_data_pairs(file_path):
    with open(file_path, "r") as f:
        pairs = json.load(f)
    return pairs

def main(args):

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base=None, 
            model_name=model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit, device_map="auto", 
            max_memory={0:"10000MiB",1:"10000MiB",2:"10000MiB",3:"10000MiB",4:"10000MiB", 5:"10000MiB", 6:"10000MiB", 7:"30000MiB"})

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

    pairs = get_data_pairs(args.file_path)
    for pair in tqdm(pairs):
        img1, cap1 = get_data_elements(pair['data_point_1'])
        img2, cap2 = get_data_elements(pair['data_point_2'])
        for j in range(2):
            conv = conv_templates[conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles
            if j == 0:
                image, true_caption, false_caption = img1, cap1, cap2
                data_entry = {"true_ann":pair['data_point_1'], "false_ann": pair['data_point_2'], "inconsistent_entity":"", "true_entity":"", "false_entity":""}
            else:
                image, true_caption, false_caption = img2, cap2, cap1
                data_entry = {"true_ann":pair['data_point_2'], "false_ann": pair['data_point_1'], "inconsistent_entity":"", "true_entity":"", "false_entity":""}
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
            inp = finetuning_datagen_prompt(roles[0], true_caption, false_caption)
            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                image = None

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = outputs
            lines = outputs.splitlines()
            keys = list(data_entry.keys())
            keys = keys[2:]
            if len(lines) == 3:
                for i in range(len(lines)):
                    line = lines[i]
                    data_entry[keys[i]] = line[line.find(": ")+len(": "):line.find(".")]
            add_result(args.save_path, annotation=data_entry)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--file_path", type=str, default="../dataset/train_pairs.json")
    parser.add_argument("--save_path", type=str, default="../dataset_ready/train.json")
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    args = parser.parse_args()
    main(args)

            
