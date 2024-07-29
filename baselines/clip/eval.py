import torch
import json
from tqdm import tqdm
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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

torch_dtype = torch.float16

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    device_map="auto",
    torch_dtype=torch_dtype,
)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def main(args):
    for data_idx in tqdm(range(args.start_idx, args.end_idx)):
        image, caption, image_path, annotation = get_data(data_idx)
        prompt1 = "The caption: {}, matches the image".format(caption)
        prompt2 = "The caption: {}, does not match the image".format(caption)
        inputs = processor(text=[prompt1, prompt2], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        results = [False, True]
        annotation['falsified'] = results[int(torch.argmax(probs))]
        add_result(args.result_file, annotation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()
    main(args)