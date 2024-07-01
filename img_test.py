from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import ollama
from transformers import AutoModel, AutoTokenizer

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

# Image vision model internlm/internlm-xcomposer2-vl-1_8b from hugging face
def model_3(image_path,prompt):
    model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b', trust_remote_code=True)
    question = f'<ImageHere> {prompt}'
    image = image_path
    response = model.chat(tokenizer, query=question, image=image, history=[], do_sample=False)
    print("Model3")
    print('Model-3 >>>>',response)
    return response

# Language mode llama3 from ollama
def llm_model(answer,res,response,final_prompt):
    
    final_response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role" : "user",
                "content" : final_prompt,
                "images" :[image_path]
            }
        ]
    )
    print("Final Result")
    print(final_response['message']['content'])


image_path = 'image/auto.jpg'
user_question = input("Enter your question Related to the Picture >>> ")
 
prompt ='describe this image and make sure to include anything notable about it (include text you see in the image)'
answer=model_1(image_path,prompt)
res=model_2(image_path,prompt)
response=model_3(image_path,prompt)

final_prompt = f"Using the following context:{answer}\n\n{res}\n\n{response}\n\nAnswer the question: {user_question}"
llm_model(answer,res,response,final_prompt)