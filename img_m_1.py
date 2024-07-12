import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

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
    
    response1 = model.answer_question(enc_image, question, tokenizer)
    print("Model_1 Response :" , response1)
    return response1
