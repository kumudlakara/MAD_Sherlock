from google.cloud import vision
import json
from PIL import Image
from newspaper import Article
import requests
from transformers import pipeline
from .data import get_data
import torch
import transformers
from transformers import AutoTokenizer
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate

def get_matching_urls(data_sample, ):
    image, caption, image_path, annotations = get_data(data_sample)
    matching_urls = []
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection


    if annotations.pages_with_matching_images:
        print("Searching the web now!")
        #print(
        #    "\n{} Pages with matching images found:".format(
        #        len(annotations.pages_with_matching_images)
        #    )
        #)

        for page in annotations.pages_with_matching_images:
            #print(f"\n\tPage url   : {page.url}")
            matching_urls.append(page.url)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return matching_urls

def get_webpage_title(matching_url):
    article = Article(matching_url)
    article.download()
    article.parse()
    return article.title

def get_webpage_text(matching_url):
    article = Article(matching_url)
    article.download()
    article.parse()
    return article.text

def get_summary(matching_urls):
    MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pipeline = transformers.pipeline("text-generation",
                        model=MODEL_NAME,
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="auto",
                        truncation=True,
                        max_length=3000,
                        max_new_tokens=1000,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id)
    llm = HuggingFacePipeline(pipeline=pipeline)
    prompt_template = """
                Write a summary of the following text delimited by triple backticks.
                Your response should cover all keypoints.
                ```{text}```
                SUMMARY:
                """
    prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    texts = []
    for matching_url in matching_urls:
        texts.append(get_webpage_text(matching_url))
    text = ""
    print("Found {} search results.".format(len(texts)))
    if len(texts) == 0:
        #to account for cases where no search results are found
        return "No search results found"
    for i in range(len(texts)):
        #only take top k=3 articles
        if i == 3: 
            break
        if "the" not in texts[i]:
            #naive way to ensure text is in English
            continue
        text += "\n\n"
        text += texts[i]
    output = llm_chain.run(text)
    return output[output.find("SUMMARY"):].rstrip()