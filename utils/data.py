import json
from PIL import Image

def get_data(i):
    visual_news_data = json.load(open("../../datasets/visualnews/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}

    data = json.load(open("../../news_clippings/news_clippings/data/merged_balanced/test.json"))
    annotations = data["annotations"]
    ann_true = annotations[i]

    caption = visual_news_data_mapping[ann_true["id"]]["caption"]
    image_path = visual_news_data_mapping[ann_true["image_id"]]["image_path"]
    image_path = "../../datasets/visualnews/origin/"+image_path[2:]
    image = Image.open(image_path)
    #print("DATA SAMPLE")
    #print("Caption: ", caption)
    #print("Misinformation (Ground Truth): {}".format(ann_true["falsified"]))
    return image, caption, image_path, ann_true

def show_data(i):
    visual_news_data = json.load(open("../../datasets/visualnews/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}

    data = json.load(open("../../news_clippings/news_clippings/data/merged_balanced/test.json"))
    annotations = data["annotations"]
    ann_true = annotations[i]

    caption = visual_news_data_mapping[ann_true["id"]]["caption"]
    image_path = visual_news_data_mapping[ann_true["image_id"]]["image_path"]
    image_path = "../../datasets/visualnews/origin/"+image_path[2:]
    image = Image.open(image_path)
    image.show()
    print("Caption: ", caption)
    print("Misinformation (Ground Truth): {}".format(ann_true["falsified"]))
