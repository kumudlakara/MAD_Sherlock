def initial_prompt(role, text):
    prompt = """{}: Given the text: {}. Does this text belong to the same context as the image or is the image being used out of context to spread misinformation?
                    The image is real. It has not been digitally altered. 
                    Carefully examine the image for any watermarks, text and other details which could tell you about the location, time or other important information to better inform your answer.
                    If you are even a little unsure of your answer or need more context, state this as UNSURE.
                    Explain your answer in detail.
                    At the end give a definite YES, NO or UNSURE answer to this question: IS THIS MISINFORMATION?""".format(role, text)
    return prompt

def initial_prompt_with_context(role, text, summary):
    prompt = """{}: This is a summary of news articles related to the image: {}
                    Based on this, you need to decide if the caption given below belongs to the image or if it is being used to spread false
                    information to mislead people.
                    CAPTION: {}
                    Note that the image is real. It has not been digitally altered. 
                    Carefully examine the image for any known entities, people, watermarks, dates, landmarks, flags, text, logos and other details which could give you important information to better explain your answer.
                    The goal is to correctly identify if this image caption pair is misinformation or not and to explain your answer in detail.
                    At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
                    """.format(role, summary, text)
    return prompt

def initial_prompt_with_context_concise(role, text, summary):
    prompt = """{}: This is a summary of news articles related to the image: {}
                    With this in mind, you need to decide if the text given below belongs with the image or if it is being used to spread false
                    information to mislead people.
                    TEXT: {}
                    Note that the image is real. It has not been digitally altered. Carefully examine the image for any known entities, watermarks, dates, landmarks, flags and other details which could tell you about the location, time or other important information to better inform your answer.
                    Give a short explanation.
                    At the end give a definite YES or NO answer to this question: IS THIS IMAGE-TEXT PAIR MISINFORMATION?""".format(role, summary, text)
    return prompt

def round1_prompt(role, text):
    prompt = """ {}: This is what I think about the same image-caption pair: {}.
                    REMEMBER: In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic.
            Do you agree with me? If you think I am wrong then convince me why.
            Clearly state your reasoning and tell me if I am missing out on some important information or am making some logical error.
            Do not describe the image. 
            At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """.format(role, text)
    return prompt

def round1_prompt_with_disambiguation(role, text):
    prompt = """ {}: Okay let's debate this. Here is what I think: {}.
            What do you think, is the image-caption pair misinformation?
            Note: don't just repeat what I am saying, you must give me new insights and a fresh perspective.
            Clearly state your reasoning and tell me if I am making some logical error.
            Do not describe the image. At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            Note: The caption does not have to describe the image exactly, they just have to be contextually related.
            ONLY if you think my argument has some ambiguities then generate 1 disambiguation query DELIMITED BY <search_query> and </search_query> TAGS, that I can use to search the web to either correct my wrong argument or strengthen my exisiting argument. 
            MAKE SURE YOU ENCLOSE THE QUERY WITHIN <search_query> and </search_query> TAGS.
            """.format(role, text)
    return prompt

def debate_prompt(role, text):
    prompt = """ {}: I see what you mean and this is what I think: {}. Do you agree with me?
                REMEMBER: In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic.
                If not then point out the inconsistencies in my argument (e.g. location, time or person related logical confusion) and explain why you are correct. 
                If you disagree with me then clearly state why and what information I am overlooking.
                Find disambiguations in my answer if any and ask questions to resolve them.
                I want you to help me improve my argument and explanation. 
                Don't give up your original opinion without clear reasons, DO NOT simply agree with me without proper reasoning.
                At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """.format(role, text)
    return prompt

def debate_prompt_with_disambiguation(role, text):
    prompt = """ {}: I see what you mean and this is what I think: {}. 
                Based on my response and your own responses so far, what do you think: is the image-caption pair misinformation or not?
                At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
                Note: The caption does not have to describe the image exactly, they just have to be contextually related.
                ONLY if you disagree with me then generate 1 disambiguation query DELIMITED BY <search_query> and </search_query> TAGS, that I can use to search the web to either correct or improve my answer. 
                MAKE SURE YOU ENCLOSE THE QUERY WITHIN <search_query> and </search_query> tags.
            """.format(role, text)
    return prompt

def refine_prompt(role, query, search_result, prev_response):
    prompt = """{}: In order to refine your earlier response, I searched the web with this query: {}. 
                This is the search result I found: {}.
                Based on this result, refine your earlier response to include more details and explanations.
                Earlier response: {}.
    """.format(role, query, search_result, prev_response)
    return prompt

def finetuning_datagen_prompt(role, true_caption, false_caption):
    #define prompt to generate template based output for data generation for finetuning
    prompt = """ {}: You are given two captions for the given image. One caption is True and the other is False.
                Based on the image, your task is to identify the most obvious element of inconsistency (MUST BE ONE WORD) such as location, time, event, person, organisation etc. in the two captions.
                TRUE CAPTION: {}
                FALSE CAPTION: {}

                You must respond by filling the following template as accurately as possible. 
                Be as concise as possble.
                <template>
                The two captions are inconsistent in: <element>. 
                The <element> in the false caption is: <false_entity>.
                However, the <element> which the true caption and image correspond to is: <true_entity>
                </template>
            """.format(role, true_caption, false_caption)
    return prompt 

def web_access_prompt(role, text):
    prompt = """ {}: You said you were UNSURE of whether the image and text belong to the same context or if they are being used to spread misinformation.
                To assist you, I did an image based internet search for the given image and found some related articles. 
                This is a summary of the articles I found online, related to the image: {}.
                Based on this new information, do you think the image and text go together or is the image being used out of context to spread misinformation?
                At the end give a definite YES, NO or UNSURE answer to this question: MISINFORMATION?
            """.format(role, text)
    return prompt

############## EXPERIMENTAL ###############
def initial_prompt_with_context_1(role, text, summary):
    prompt = """{}: This is a summary of text related to the image taken from a news article: {}
    REMEMBER: In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic. The image may be related to the news article and caption, yet not depict them exactly.
Based on the given summary and the image provided, you need to give an explanation for why the caption given below when used with the image, is being used in a way that's consistent with the content of the news article.
CAPTION: {}
Note that the image itself is real and has not been digitally altered. It is taken from a genuine news article.
Carefully examine the image for any known entities, people, watermarks, dates, landmarks, flags, text, logos, and other details which could provide important contextual information.
Your task is to reason why the image-caption pair, when considered alongside the article summary, presents accurate information.
Explain your reasoning in detail, considering all aspects of the information provided.
                    """.format(role, summary, text)
    return prompt

def initial_prompt_with_context_2(role, text, summary):
    prompt = """{}: This is a summary of text related to the image taken from a news article: {}
    REMEMBER: In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic. The image may be related to the news article and caption, yet not depict them exactly.
Based on the given summary and the image provided, you need to explain why the caption given below when used with the image, is being used to spread false information and mislead people.
CAPTION: {}
Note that the image itself is real and has not been digitally altered. It is taken from a genuine news article.
Carefully examine the image for any known entities, people, watermarks, dates, landmarks, flags, text, logos, and other details which could provide important contextual information.
Your task is to clearly explain why the image-caption pair, when considered alongside the article summary, is misinformation. 
Explain your reasoning in detail, considering all aspects of the information provided.
                    """.format(role, summary, text)
    return prompt

############## ACTOR-SKEPTIC SETUP############
def initial_prompt_actor(role, text, summary):
    prompt = """{}: This is a summary of text related to the image taken from a news article: {}
Based on this summary and the image provided, you need to determine if the caption given below when used with the image, is being used in a way that's consistent with the content of the news article, or if it's being used to spread false information and mislead people.
CAPTION: {}
Note that the image itself is real and has not been digitally altered. It is taken from a genuine news article.
Carefully examine the image for any known entities, people, watermarks, dates, landmarks, flags, text, logos, and other details which could provide important contextual information.
Your task is to assess whether the image-caption pair, when considered alongside the article summary, presents accurate information or if it's being used in a misleading way. 
Explain your reasoning in detail, considering all aspects of the information provided.
At the end of your analysis, provide a definite YES or NO answer to this question: IS THIS MISINFORMATION?
Remember: In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic.
                    """.format(role, summary, text)
    return prompt

def initial_prompt_skeptic(role, actor_response):
    prompt = """{}: You are a skeptical AI agent tasked with critically evaluating responses and uncovering potential flaws or inconsistencies. Your goal is to formulate a single, targeted question that challenges or clarifies the reasoning presented.
Context: I was given an image from a news article along with a caption. My task was to determine if the caption genuinely belongs to the image and news article, or if it's being used to spread misinformation. Here's their response:
{}
Your task:

Carefully analyze the given response for any potential logical flaws, unsupported assumptions, or areas lacking clarity.
Identify the most crucial aspect of the response that requires further scrutiny or elaboration.
Formulate ONE specific, probing question that directly addresses this aspect.

Your question should:

Be directly related to the details or reasoning provided in the response
Target any ambiguities, inconsistencies, or potential weaknesses in the argument
Aim to reveal any gaps in logic or prompt the agent to provide additional evidence for their claims
Be precise and focused, avoiding broad or general inquiries
DO NOT make up scenarios such as "if .."

Remember, your role is not to determine if the original assessment is correct or incorrect, but to challenge the reasoning process and push for greater clarity and depth of analysis.
                """.format(role, actor_response)
    return prompt


def refine_actor_response_prompt(role, text):
    prompt = """ {}: I am skeptical of what your response. Here is what I have to say: {}. 
                Based on my apprehensions, do you still think you are correct? 
                If required, correct yourself, otherwise refine your response to be more detailed by answering any questions/apprehensions I have.
                At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """.format(role, text)
    return prompt

def end_decision_prompt(role, prev, text):
    prompt = """{}: I took into account your suggestions and refined my response. 
            Earlier my answer to the question: IS THIS MISINFORMATION? was: {}. 
            My new response after taking into account your skepticism is: {}
            If there has been a complete reversal in my response (e.g., from 'Yes, this is misinformation' to 'No, this is not misinformation' or vice versa), be extra cautious and critical in your evaluation and ask me to 'CONTINUE' my analysis.
            If you think my new response is correct and I don't need to refine it further then output 'PERFECT'.
            If you think I should refine my response further and I may be wrong then output: 'CONTINUE'.
            """.format(role, prev, text)
    return prompt

############## EXPERIMENTAL PROMPT REFINEMENT FOR FINETUNING #############
def initial_prompt_with_context_better(role, text, summary):
    prompt = """{}: This is a summary of text related to the image taken from a news article: {}
    REMEMBER: In news articles, captions often don't exactly describe the image (but are still related to the image) but are contextually related to the broader story. Focus on whether the image-caption pair, in conjunction with the article summary, presents an accurate representation of the news event or topic. The image may be related to the news article and caption, yet not depict them exactly.
Based on the given summary and the image provided, you need to determine if the caption given below when used with the image, is being used in a way that's consistent with the content of the news article, or if it's being used to spread false information and mislead people.
CAPTION: {}
Note that the image itself is real and has not been digitally altered. It is taken from a genuine news article.
Carefully examine the image for any known entities, people, watermarks, dates, landmarks, flags, text, logos, and other details which could provide important contextual information.
Your task is to assess whether the image-caption pair, when considered alongside the article summary, presents accurate information or if it's being used in a misleading way. 
Explain your reasoning in detail, considering all aspects of the information provided.
At the end of your analysis, provide a definite YES or NO answer to this question: IS THIS MISINFORMATION?
                    """.format(role, summary, text)
    return prompt
