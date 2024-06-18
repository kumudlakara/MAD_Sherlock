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
                    With this in mind, you need to decide if the caption given below belongs to the image or if it is being used to spread false
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

def web_access_prompt(role, text):
    prompt = """ {}: You said you were UNSURE of whether the image and text belong to the same context or if they are being used to spread misinformation.
                To assist you, I did an image based internet search for the given image and found some related articles. 
                This is a summary of the articles I found online, related to the image: {}.
                Based on this new information, do you think the image and text go together or is the image being used out of context to spread misinformation?
                At the end give a definite YES, NO or UNSURE answer to this question: MISINFORMATION?
            """.format(role, text)
    return prompt