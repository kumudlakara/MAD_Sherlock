from google.cloud import vision
import json
from PIL import Image
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from .data import get_data

def get_matching_urls(data_sample, ):
    image, caption, image_path = get_data(data_sample)
    matching_urls = []
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection


    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )

        for page in annotations.pages_with_matching_images:
            print(f"\n\tPage url   : {page.url}")
            matching_urls.append(page.url)

            if page.full_matching_images:
                print(
                    "\t{} Full Matches found: ".format(len(page.full_matching_images))
                )

                for image in page.full_matching_images:
                    print(f"\t\tImage url  : {image.url}")

            if page.partial_matching_images:
                print(
                    "\t{} Partial Matches found: ".format(
                        len(page.partial_matching_images)
                    )
                )

                for image in page.partial_matching_images:
                    print(f"\t\tImage url  : {image.url}")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return matching_urls

def get_webpage_title(matching_url):
    request = requests.get(matching_url)
    soup = BeautifulSoup(request.text, "html.parser")
    return soup.title.string

def get_webpage_text(matching_url):
    request = requests.get(matching_url)
    soup = BeautifulSoup(request.text, "html.parser")
    return soup.text