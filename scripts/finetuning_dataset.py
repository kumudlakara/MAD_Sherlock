import sys
sys.path.append("..")

import numpy as np
import json
from PIL import Image
from tqdm import tqdm

from utils.compile_results import add_result

search_dirs = ["../../news_clippings/news_clippings/data/merged_balanced/train.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/train.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/train.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/train.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/train.json",
 "../../news_clippings/news_clippings/data/merged_balanced/test.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/test.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/test.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/test.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/test.json",
 "../../news_clippings/news_clippings/data/merged_balanced/val.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/val.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/val.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/val.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/val.json"]

def get_data(file_path, i):
    data = json.load(open(file_path))
    annotations = data["annotations"]
    ann1 = annotations[i]
    ann2 = annotations[i+1]
    return ann1, ann2

def find_true(search_dirs, id_to_find):
    for search_dir in search_dirs:
        temp_data = json.load(open(search_dir))
        annotations = temp_data["annotations"]
            #search for true
        for ann in annotations:
            if id_to_find == ann['id'] and ann['falsified'] == False:
                return search_dir, ann

    return ";-;", ":("


file_path = "../../news_clippings/news_clippings/data/merged_balanced/train.json"
save_file = "../dataset/train.json"
for i in tqdm(range(0, 71072, 2)):
    ann1, ann2 = get_data(file_path, i)
    id_to_find = ann2['image_id']
    search_dir, ann2_true = find_true(search_dirs, id_to_find)
    new_data_point = {"data_point_1":ann1, "data_point_2":ann2_true, "search_dir":search_dir}
    add_result(save_file, new_data_point)

print("Training data done!")


search_dirs = [
 "../../news_clippings/news_clippings/data/merged_balanced/test.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/test.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/test.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/test.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/test.json",
 "../../news_clippings/news_clippings/data/merged_balanced/train.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/train.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/train.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/train.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/train.json",
 "../../news_clippings/news_clippings/data/merged_balanced/val.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/val.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/val.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/val.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/val.json"]
file_path = "../../news_clippings/news_clippings/data/merged_balanced/test.json"
save_file = "../dataset/test.json"
for i in tqdm(range(0, 7264, 2)):
    ann1, ann2 = get_data(file_path, i)
    id_to_find = ann2['image_id']
    search_dir, ann2_true = find_true(search_dirs, id_to_find)
    new_data_point = {"data_point_1":ann1, "data_point_2":ann2_true, "search_dir":search_dir}
    add_result(save_file, new_data_point)

print("Testing data done!")

search_dirs = [ "../../news_clippings/news_clippings/data/merged_balanced/val.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/val.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/val.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/val.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/val.json",
 "../../news_clippings/news_clippings/data/merged_balanced/test.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/test.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/test.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/test.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/test.json",
 "../../news_clippings/news_clippings/data/merged_balanced/train.json", 
 "../../news_clippings/news_clippings/data/person_sbert_text_text/train.json",
 "../../news_clippings/news_clippings/data/scene_resnet_place/train.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_image/train.json",
 "../../news_clippings/news_clippings/data/semantics_clip_text_text/train.json",
]
file_path = "../../news_clippings/news_clippings/data/merged_balanced/val.json"
save_file = "../dataset/val.json"
for i in tqdm(range(0, 7024, 2)):
    ann1, ann2 = get_data(file_path, i)
    id_to_find = ann2['image_id']
    search_dir, ann2_true = find_true(search_dirs, id_to_find)
    new_data_point = {"data_point_1":ann1, "data_point_2":ann2_true, "search_dir":search_dir}
    add_result(save_file, new_data_point)
print("Validation data done!")