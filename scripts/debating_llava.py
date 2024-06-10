import sys
sys.path.append('..')

import torch
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from tqdm import tqdm
import argparse
from utils.data import get_data, show_data
from utils.compile_results import add_result

def initial_prompt(role, text):
    prompt = """{}: Given the text: {}. Does this text belong to the same context as the image or is the image being used out of context to spread misinformation?
                    The image is real. It has not been digitally altered. 
                    Carefully examine the image for any watermarks, text and other details which could tell you about the location, time or other important information to better inform your answer.
                    Explain your answer in detail. 
                    At the end give a definite YES or NO or UNSURE answer to this question: MISINFORMATION?""".format(role, text)
    return prompt

def round1_prompt(role, text):
    prompt = """ {}: This is what I think: {}. Do you agree with me? If you think I am wrong then convince me.
            Clearly state your reasoning and tell me if I am missing out on some important information or am making some logical error.
            Do not describe the image. 
            At the end give a definite YES or NO answer to this question: MISINFORMATION?
            """.format(role, text)
    return prompt

def debate_prompt(role, text):
    prompt = """ {}: I see what you mean and this is what I think: {}. Do you agree with me?
                If not then point out the inconsistencies in my argument (e.g. location, time or person related logical confusion) and explain why you are correct. 
                If you disagree with me then clearly state why and what information I am overlooking.
                You can also ask me rebutal questions to find loopholes in my reasoning. 
                Don't give up your original opinion without clear reasons, DO NOT simply agree with me without proper reasoning.
                At the end give a definite YES or NO answer to this question: MISINFORMATION?
            """.format(role, text)
    return prompt

def get_final_prediction(num_models, model_responses):
    num_true, num_false = 0,0
    final_pred = ""
    final_outputs = {}
    for i in range(num_models):
        if model_responses[i]["falsified"] == True:
            num_true += 1
        else:
            num_false += 1
        final_outputs["model_"+str(i)] = model_responses[i]["output"]

    if num_true > num_false:
        final_pred = "True"
    else:
        final_pred = "False"
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

def main(args):
    models = []
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    for i in range(args.num_models):
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base=None, model_name=model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit, device_map="auto")
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
    for data_idx in tqdm(range(7264)):
        conv, roles = get_conv_and_roles(model_name, conv_mode)
        image, caption, img_path, annotation = get_data(data_idx)
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
        for round in range(args.num_rounds+1):
            for i in range(args.num_models):
                if round == 0:
                    inp = initial_prompt(roles[i][0], caption)
                    #print("INPUT MESSAGE: ", inp)
                elif round == 1:
                    if i == 1:
                        inp = round1_prompt(roles[i][0], temp)
                        #print("INPUT MESSAGE: ", inp)
                    else:
                        inp = round1_prompt(roles[i][0], conv[(i+1)%args.num_models].messages[-1][-1])
                else:
                    inp = debate_prompt(roles[i][0], conv[(i+1)%args.num_models].messages[-1][-1])
                    #print("INPUT MESSAGE: ", inp)
                #print("========================== Agent - {} =====================".format(i+1))
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
                prompt = conv[i].get_prompt()

                input_ids = tokenizer_image_token(prompt, models[i]['tokenizer'], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(models[i]['model'].device)
                stop_str = conv[i].sep if conv[i].sep_style != SeparatorStyle.TWO else conv[i].sep2
                keywords = [stop_str]
                #streamer = TextStreamer(models[i]['tokenizer'], skip_prompt=True, skip_special_tokens=True)

                with torch.inference_mode():
                    output_ids = models[i]['model'].generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image_size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True)

                outputs = models[i]['tokenizer'].decode(output_ids[0]).strip()
                conv[i].messages[-1][-1] = outputs
                if i == 0 and round == 0:
                    temp = outputs
                if round == args.num_rounds-1:
                    #final answer from the model
                    if "YES" in outputs:
                        model_responses[i]["falsified"] = True
                        model_responses[i]["output"] = outputs
                    elif "NO" in outputs:
                        model_responses[i]["falsified"] = False
                        model_responses[i]["output"] = outputs

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
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    args = parser.parse_args()
    main(args)