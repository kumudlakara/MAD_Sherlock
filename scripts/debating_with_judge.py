"""
Models always have web context to begin with.
"""

import sys
sys.path.append('..')

import torch
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from peft import PeftModel

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate

from tqdm import tqdm
import argparse
import json

from utils.data import get_data
from utils.compile_results import add_result
from utils.prompts import debater_prompt, judge_prompt


def get_debater_conv(name, word_limit, answer_defending, answer_opposing, caption, summary):
    sys_msg = """<|im_start|>system\nYou will act as an expert debater named {NAME} engaging in a debate about whether the given image when considered alongside the below caption is misinformation. You will be shown two possible answers to the question "Is the image-caption pair misinformation?". You are also given a summary of news articles in which the image actually appears. You will be assigned to argue in favour of one answer while another debater will defend the other. A judge, unable to see the image, caption and new articles, will decide the winner. Your goal is to persuade the judge that your answer is correct.
        CAPTION: {CAPTION}
        SUMMARY: {SUMMARY}
      You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. Do not exceed {WORD_LIMIT} words per argument. Please do not include an ”In summary”, ”In conclusion”, or equivalent paragraph at the end of your argument because more debate rounds will follow. Structure your output as follows:
     <argument>[ARGUMENT]</argument> The judge only sees the output in <argument> tags, so focus on clarity and persuasiveness in this section.
    Judging Criteria:
    • Considering the contextual relationship between the image, caption and news article summary is crucial, and how you integrate them into a logical and compelling argument is essential.
    • Deductions for unverified claims and repeating arguments.
    • Points for developing arguments further and introducing new ones.
    • Points for demonstrating misleading claims from your opponent and for convincing counter arguments.""".format(NAME=name, WORD_LIMIT=word_limit, CAPTION=caption, SUMMARY=summary)

    msgs = [("<|im_start|>user\n","""Here is the question and the two answers:
    <question>Based on context, is the image-caption pair misinformation?</question>
    <your answer>{ANSWER_DEFENDING}</your answer>
    <opponent answer>{ANSWER_OPPOSING}</opponent answer>
    Please restate your position and the position you are arguing against to confirm you understand.""".format(ANSWER_DEFENDING=answer_defending, ANSWER_OPPOSING=answer_opposing)),
    ("<|im_start|>assistant\n","""I understand. I will argue that the answer to the question ”Based on context, is the image-caption pair misinformation?” is ”{ANSWER_DEFENDING}”. I will argue against the answer ”{ANSWER_OPPOSING}”.""".format(ANSWER_DEFENDING=answer_defending, ANSWER_OPPOSING=answer_opposing))
    ]

    conv_template = Conversation(
        system=sys_msg,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        version="mpt",
        messages=msgs,
        offset=0,
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
    return conv_template

def get_thinking_advice(round):
    advice = [
        """ 1. Consider what extra context the judge needs to know about the image, caption and news articles for your answer to make sense. You should include this context in your argument.
            2. Thinking step by step,plan an argument for your position.
            3. Select the best points from the summary (if possible) to use in your argument.
        """,
        """ 1. List out the biggest flaws in your opponent’s argument.
            2. Thinking step by step, critique your opponent’s argument.
            3. Select the best points from the summary (if possible) to disprove your opponent’s argument.
        """,
        """ 1. List the critiques the opponent has made to your argument.
            2. Thinking step by step, construct the best counter argument to your opponent’s critique.
            3. Select the best points from the summary(if possible) to disprove your opponent’s critique.
        """
    ]
    return advice[round]

def get_new_arg_request(round):
    request = [
        """Now it’s your turn to construct your opening argument for why the answer to the question ”Is the image-caption pair misinformation?” is ”[ANSWER DEFENDING]”.""",
        """Now it’s your turn to construct your next argument for why the answer to the question ”Is the image-caption pair misinformation?” is ”[ANSWER DEFENDING]”."""
    ]
    if round == 0:
        return request[0]   
    else:
        return request[1]

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

def retrieve_summary(key):
    with open("../utils/summaries.json", "r") as f:
        data = json.load(f)
    return data[key]

def main(args):
    """
    llm_name = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    pipeline = transformers.pipeline(
                        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=1000,
        do_sample=True)
    llm = HuggingFacePipeline(pipeline=pipeline)

    """
    models = []
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    for i in range(args.num_models+1):
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base=None, model_name=model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit, device_map="auto")
        if args.load_finetuned:
            model = PeftModel.from_pretrained(model, args.finetuned_model_path, device_map="auto")
            model.merge_and_unload()
            model.to(dtype=torch.bfloat16)
        models.append({"tokenizer":tokenizer, "model":model, "image_processor":image_processor, "context_len":context_len})

    answer_a = "No, the image caption pair is not misinformation."
    answer_b = "Yes, the image caption pair is misinformation."

    print("Running inference now!")
    for data_idx in tqdm(range(args.start_idx,args.end_idx)):
        image, caption, img_path, annotation = get_data(data_idx)
        output_a, output_b = "<argument> ","<argument> "
        summary_key = str(annotation['id'])+"_"+str(annotation["image_id"])
        summary = retrieve_summary(summary_key)
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], models[0]['image_processor'], models[0]['model'].config)
        if type(image_tensor) is list:
            image_tensor = [image.to(models[0]['model'].device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(models[0]['model'].device, dtype=torch.float16)
        conv_a = get_debater_conv("DEBATER-A", args.word_limit, answer_a, answer_b, caption, summary)
        roles_a = conv_a.roles
        conv_b = get_debater_conv("DEBATER-B", args.word_limit, answer_b, answer_a, caption, summary)
        roles_b = conv_b.roles

        
        transcript = ""
        output = ""
        for round in range(args.num_rounds):
            #conv, roles = get_conv_roles()
            thinking_advice = get_thinking_advice(round)
            new_arg_request = get_new_arg_request(round)
            
            new_arg_req_a = new_arg_request.replace("[ANSWER DEFENDING]", answer_a)
            new_arg_req_b = new_arg_request.replace("[ANSWER DEFENDING]", answer_b)

            debater_prompt_a = debater_prompt(roles_a, args.word_limit, answer_a, answer_b, output_b[output_b.find("<argument>")+len("<argument>"):], new_arg_req_a, thinking_advice)
            debater_prompt_b = debater_prompt(roles_b, args.word_limit, answer_b, answer_a, output_a[output_a.find("<argument>")+len("<argument>"):], new_arg_req_b, thinking_advice)
            
            if round == 0:
                # first message
                if models[0]['model'].config.mm_use_im_start_end:
                    debater_prompt_a = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + debater_prompt_a
                    debater_prompt_b = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + debater_prompt_b
                else:
                    debater_prompt_a = DEFAULT_IMAGE_TOKEN + '\n' + debater_prompt_a
                    debater_prompt_b = DEFAULT_IMAGE_TOKEN + '\n' + debater_prompt_b
            
            
            conv_a.append_message(conv_a.roles[0], debater_prompt_a)
            conv_a.append_message(conv_a.roles[1], None)
            output_a = generate_output(0, [conv_a,conv_b], models, image_tensor, args.temperature, image_size, args.max_new_tokens)
            conv_a.messages[-1][-1] = output_a[output_a.find("<argument>")+len("<argument>"):]
            conv_b.append_message(conv_b.roles[0], debater_prompt_b)
            conv_b.append_message(conv_b.roles[1], None)
            output_b = generate_output(1, [conv_a, conv_b], models, image_tensor, args.temperature, image_size, args.max_new_tokens)
            conv_b.messages[-1][-1] = output_b[output_b.find("<argument>")+len("<argument>"):]
            if '<argument>' in output_a:
                transcript += "\nDEBATER-A:{}\nDEBATER-B:{}\n".format(output_a[output_a.find("<argument>")+len("<argument>"):], output_b[output_b.find("<argument>")+len("<argument>"):])
            else:
                transcript += "\nDEBATER-A:{}\nDEBATER-B:{}\n".format(output_a, output_b)
            transcript = transcript.lstrip().rstrip()
            while "  " in transcript:
                transcript = transcript.replace("  ", " ")
        transcript = transcript.replace("</argument>","")
        transcript = transcript.replace("<|im_end|>", "")           
        judge_conv = conv_templates["chatml_direct"].copy()
        judge_roles = judge_conv.roles
        prompt = """{ROLE}: You are an expert judge in a debate where the participants are arguing over whether a given image-caption pair is misinformation or not, based on a summary of news articles. Your task is to determine which debater is correct. However, you dont have access to the news articles, making your task challenging. To guide you, debaters can include direct details of the image, caption and news articles. Your main objective is to choose the correct answer based on the evidence presented. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating arguments.
Here is the caption: {CAPTION}
Here is the question and the two answers:
Question: Is the image-caption pair misinformation?
A: {ANSWER_A}
B: {ANSWER_B}
{NAME_A} is arguing for answer A, and {NAME_B} is arguing for answer B.
Here’s the current transcript of the debate:
<transcript>
{TRANSCRIPT}
</transcript>
In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic. The image may be related to the news article and caption, yet not depict them exactly.
Choose the answer you think is most likely to be correct based on the transcript. Think about your answer in <thinking></thinking> tags in 100 words by carrying out these steps:
1. Think about the strengths and weaknesses of each debater’s arguments and whether you can trust them.
2. Think step by step about which answer is likely to be correct. 
On new line, format your final answer as ”Answer: <A|B>”. At the end you MUST pick either A or B""".format(ROLE=judge_roles[0], CAPTION=caption, TRANSCRIPT=transcript, ANSWER_A=answer_a, ANSWER_B=answer_b, NAME_A="DEBATER-A", NAME_B="DEBATER-B")
        judge_inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        judge_conv.append_message(judge_conv.roles[0], judge_inp)
        judge_conv.append_message(judge_conv.roles[1], None)
        output = generate_output(2, ["","",judge_conv], models, image_tensor, args.temperature, image_size, args.max_new_tokens)
        count = 0
        while count < 5:   
            #try to get output in proper format
            if "Answer: A" in output or "ANSWER: A" in output:
                annotation['falsified'] = False
                break
            if "Answer: B" in output or "ANSWER: B" in output:
                annotation['falsified'] = True
                break
            count += 1
            judge_conv = conv_templates["chatml_direct"].copy()
            judge_roles = judge_conv.roles
            prompt = """{ROLE}: You are an expert judge in a debate where the participants are arguing over whether a given image-caption pair is misinformation or not, based on a summary of news articles. Your task is to determine which debater is correct. However, you dont have access to the news articles, making your task challenging. To guide you, debaters can include direct details of the image, caption and news articles. Your main objective is to choose the correct answer based on the evidence presented. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating arguments.
Here is the caption: {CAPTION}
Here is the question and the two answers:
Question: Is the image-caption pair misinformation?
A: {ANSWER_A}
B: {ANSWER_B}
{NAME_A} is arguing for answer A, and {NAME_B} is arguing for answer B.
Here’s the current transcript of the debate:
<transcript>
{TRANSCRIPT}
</transcript>
In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic. The image may be related to the news article and caption, yet not depict them exactly.
Choose the answer you think is most likely to be correct based on the transcript. Think about your answer in <thinking></thinking> tags in 100 words by carrying out these steps:
1. Think about the strengths and weaknesses of each debater’s arguments and whether you can trust them.
2. Think step by step about which answer is likely to be correct. 
On new line, format your final answer as ”Answer: <A|B>”. At the end you MUST pick either A or B""".format(ROLE=judge_roles[0], CAPTION=caption, TRANSCRIPT=transcript, ANSWER_A=answer_a, ANSWER_B=answer_b, NAME_A="DEBATER-A", NAME_B="DEBATER-B")
            judge_inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            judge_conv.append_message(judge_conv.roles[0], judge_inp)
            judge_conv.append_message(judge_conv.roles[1], None)
            output = generate_output(2, ["","",judge_conv], models, image_tensor, args.temperature, image_size, args.max_new_tokens)
        annotation['output'] = transcript+"\n\n"+output
        add_result(args.result_file, annotation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--load_finetuned", type=bool, default=False)
    parser.add_argument("--finetuned_model_path", type=str, default="../../datasets/models/checkpoints/llava-v1_6_34b_finetuning_2/checkpoint-6000/")
    parser.add_argument("--num_models", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--word_limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--result_file", type=str, default="results.json")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    args = parser.parse_args()
    main(args)
