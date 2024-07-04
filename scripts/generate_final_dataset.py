import sys
sys.path.append("..")
import json
from tqdm import tqdm
import argparse

from utils.data import get_data_elements


def get_img_path(ann):
    visual_news_data = json.load(open("../../datasets/visualnews/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}


    caption = visual_news_data_mapping[ann["id"]]["caption"]
    image_path = visual_news_data_mapping[ann["image_id"]]["image_path"]
    image_path = "./visualnews/origin/"+image_path[2:]
    return image_path


def main(args):
    with open(args.load_path, "r") as f:
        data = json.load(f)
    ready_dataset = []
    for dp in tqdm(data):
        true_ann, false_ann = dp['true_ann'], dp['false_ann']
        data_dict = {}
        _, true_cap = get_data_elements(true_ann)
        _, false_cap = get_data_elements(false_ann)
        data_dict["id"] = str(true_ann["id"])+"_"+str(false_ann['id'])
        data_dict["image"] = get_img_path(dp['true_ann'])
        data_dict["conversations"] = [
        {
            "from": "human",
            "value": """<image>\nYou are given two captions for the given image. One caption is True and the other is False.
                    Based on the image, your task is to identify the most obvious element of inconsistency (MUST BE ONE WORD) such as location, time, event, person, organisation etc. in the two captions.
                    TRUE CAPTION: {}
                    FALSE CAPTION: {}

                    You must respond by filling the following template as accurately as possible. 
                    Be as concise as possble.
                    <template>
                    The two captions are inconsistent in: <element>. 
                    The <element> in the false caption is: <false_entity>.
                    However, the <element> which the true caption and image correspond to is: <true_entity>
                    </template>""".format(true_cap, false_cap)
        },
        {
            "from": "gpt",
            "value": """The two captions are inconsistent in: {}. 
                    The {} in the false caption is: {}.
                    However, the {} which the true caption and image correspond to is: {}""".format(dp['inconsistent_entity'], dp['inconsistent_entity'],dp['false_entity'],dp['inconsistent_entity'], dp['true_entity'])
        },
        ]
        ready_dataset.append(data_dict)
    
    with open(args.save_path, "w") as f:
        json.dump(ready_dataset, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../../datasets/finetuning_dataset/pre_format/val.json")
    parser.add_argument("--save_path", type=str, default="../../datasets/finetuning_dataset/val.json")
    args = parser.parse_args()
    main(args)