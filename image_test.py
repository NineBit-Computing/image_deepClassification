from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import ollama
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List


# Image vision model vikhyatk/moondream2 from hugging face
def model_1(image_path, prompt):
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    image = Image.open(image_path)
    question = prompt
    enc_image = model.encode_image(image)
    
    answer = model.answer_question(enc_image, question, tokenizer)
    print("Model1")
    print('model-1',answer)
    return answer

# Mult-model llava from ollama
def model_2(image_path,prompt):
    res = ollama.chat(
        model="llava",
        messages=[
            {
                "role" : "user",
                "content" : prompt,
                "images" :[image_path]
            }
        ]
    )
    print("Model2")
    print(res['message']['content'])
    return res

# Language mode llama3 from ollama
def llm_model(answer,res,response,final_prompt):
    
    final_response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role" : "user",
                "content" : final_prompt,
            }
        ]
    )
    print("Final Result")
    print(final_response['message']['content'])
    return final_response['message']['content']




