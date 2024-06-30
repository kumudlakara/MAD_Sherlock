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
                    Note that the image is real. It has not been digitally altered. Carefully examine the image for any known entities, watermarks, dates, landmarks, flags and other details which could tell you about the location, time or other important information to better inform your answer.
                    Explain your answer in detail.
                    At the end give a definite YES or NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?""".format(role, summary, text)
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
    prompt = """ {}: This is what I think: {}. Do you agree with me? If you think I am wrong then convince me why you are correct.
            Clearly state your reasoning and tell me if I am missing out on some important information or am making some logical error.
            Do not describe the image. 
            At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """.format(role, text)
    return prompt

def round1_prompt_with_disambiguation(role, text):
    prompt = """ {}: This is what I think: {}. Do you agree with me? If you think I am wrong then convince me why you are correct.
            Clearly state your reasoning and tell me if I am missing out on some important information or am making some logical error.
            Do not describe the image. At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            Note: The caption does not have to describe the image exactly, they just have to be contextually related.
            ONLY if you think my argument has some ambiguities then generate 1 disambiguation query DELIMITED BY <search_query> and </search_query> TAGS, that I can use to search the web to either correct my wrong argument or strengthen my exisiting argument. 
            MAKE SURE YOU ENCLOSE THE QUERY WITHIN <search_query> and </search_query> TAGS.
            """.format(role, text)
    return prompt

def debate_prompt(role, text):
    prompt = """ {}: I see what you mean and this is what I think: {}. Do you agree with me?
                If not then point out the inconsistencies in my argument (e.g. location, time or person related logical confusion) and explain why you are correct. 
                If you disagree with me then clearly state why and what information I am overlooking.
                Find disambiguations in my answer if any and ask questions to resolve them.
                I want you to help me improve my argument and explanation. 
                Don't give up your original opinion without clear reasons, DO NOT simply agree with me without proper reasoning.
                At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """.format(role, text)
    return prompt

def debate_prompt_with_disambiguation(role, text):
    prompt = """ {}: I see what you mean and this is what I think: {}. Do you agree with me?
                If not then point out the inconsistencies in my argument (e.g. location, time or person related logical confusion) and explain why you are correct. 
                If you disagree with me then clearly state why and what information I am overlooking.
                At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
                Note: The caption does not have to describe the image exactly, they just have to be contextually related.
                ONLY if you think my response has some ambiguities or missing information then generate 1 disambiguation query DELIMITED BY <search_query> and </search_query> TAGS, that I can use to search the web to either correct my wrong argument or strengthen my exisiting argument. 
                MAKE SURE YOU ENCLOSE THE QUERY WITHIN <search_query> and </search_query> tags.
            """.format(role, text)
    return prompt

def refine_prompt(role, query, search_result, prev_response):
    prompt = """{}: In order to refine your earlier response, I searched the web with this query: {}. 
                This is the search result I found: {}.
                Based on this result, refine your earlier response to include more details and explanations.
                Remove the disambiguation query if there is one demilited by <search_query> tags.
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
    prompt = """{}: This is a summary of news articles related to the image: {}
                    Based on this summary, tell me why the caption belongs with the image.
                    Note the image does not need to show exactly what the caption mentions, they just have to be related.
                    CAPTION: {}
                    Note that the image is real and the caption is correct. It has not been digitally altered. Carefully examine the image for any known entities, watermarks, dates, landmarks, flags and other details which could tell you about the location, time or other important information to better inform your answer.
                    Explain your answer in detail.
                    At the end give a definite NO answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?""".format(role, summary, text)
    return prompt

def initial_prompt_with_context_2(role, text, summary):
    prompt = """{}: This is a summary of news articles related to the image: {}
                    Based on this summary, you must argue that the caption given below does NOT belong to the image and IS being used to spread false
                    information to mislead people.
                    Note that the image does not need to show exactly what the caption mentions, they just have to be related.
                    CAPTION: {}
                    Note that the image is real and caption is false. It has not been digitally altered. Carefully examine the image for any known entities, watermarks, dates, landmarks, flags and other details which could tell you about the location, time or other important information to better inform your answer.
                    Explain your answer in detail.
                    At the end give a definite YES answer to this question: IS THIS IMAGE-CAPTION PAIR MISINFORMATION?""".format(role, summary, text)
    return prompt


