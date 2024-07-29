import sys
sys.path.append("..")

from IPython.display import Image, display
from PIL import Image
import io
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
import json
import argparse
from tqdm import tqdm
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast

URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

objids = utils.get_data(OBJ_URL)
attrids = utils.get_data(ATTR_URL)
vqa_answers = utils.get_data(VQA_URL)

def get_data(i):
    visual_news_data = json.load(open("../../../datasets/visualnews/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}

    data = json.load(open("../../../news_clippings/news_clippings/data/merged_balanced/test.json"))
    annotations = data["annotations"]
    ann_true = annotations[i]

    caption = visual_news_data_mapping[ann_true["id"]]["caption"]
    image_path = visual_news_data_mapping[ann_true["image_id"]]["image_path"]
    image_path = "../../../datasets/visualnews/origin/"+image_path[2:]
    image = Image.open(image_path)
    #print("DATA SAMPLE")
    #print("Caption: ", caption)
    #print("Misinformation (Ground Truth): {}".format(ann_true["falsified"]))
    return image, caption, image_path, ann_true

def add_result(file_path, annotation):
    try:
        # Read the existing file
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        # If the file doesn't exist or is empty, initialize an empty list
        data = []

    # Append the new result to the list
    data.append(annotation)

    # Write the updated list to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def main(args):
    #load models and components
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

    image_preprocess = Preprocess(frcnn_cfg)

    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

    for data_idx in tqdm(range(args.start_idx, args.end_idx)):
        image, caption, image_path, annotation = get_data(data_idx)
        images, sizes, scales_yx = image_preprocess(image_path)
        output_dict = frcnn(
                        images,
                        sizes,
                        scales_yx=scales_yx,
                        padding="max_detections",
                        max_detections=frcnn_cfg.max_detections,
                        return_tensors="pt",
                    )
        features = output_dict.get("roi_features")
        prompt = "Does this caption: {}, match the image?".format(caption)
        inputs = bert_tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                        return_token_type_ids=True,
                        return_attention_mask=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
        output_vqa = visualbert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_embeds=features,
            visual_attention_mask=torch.ones(features.shape[:-1]),
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )
        pred = output_vqa["logits"].argmax(-1)
        answer = vqa_answers[pred]
        if answer == "yes":
            annotation['falsified'] = False
        else:
            annotation['falsified'] = True
        
        add_result(args.result_file, annotation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
