from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import image_test
import os
app = FastAPI()

class Operator(BaseModel):
    taskName: str

class ModelDetail(BaseModel):
    promptFromUser: str
    objectName: str
    operators: List[Operator]

@app.post("/run-model")
async def handle_image_prompt(modelDetail: ModelDetail):
    image_path = f'/home/bharat/codebase/unstr_to_str/image/{modelDetail.objectName}'
    user_question = modelDetail.promptFromUser

    
    prompt = 'describe this image and make sure to include anything notable about it (include text you see in the image)'

    results = {}

    # Execute each operator's task
    for operator in modelDetail.operators:
        task_name = operator.taskName
        model_func = getattr( image_test, task_name)
        
        results[task_name] = model_func(image_path, prompt)
            

    final_prompt = f"Given the context derived from analyzing the image, provide a comprehensive answer to the user's question.\n\nContext:\n"
    for key, value in results.items():
        final_prompt += f"{key}: {value}\n\n"
    final_prompt += f"Answer the question: {user_question}"

    final_response = image_test.llm_model(
        results.get("model_1"),
        results.get("model_2"),
        results.get("model_3"),
        final_prompt,
    )
    
    return {"final_response": final_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
