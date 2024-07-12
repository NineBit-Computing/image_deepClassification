import img_m_1
import img_m_2
from rev_engg import rephrase_model
from chat_model import final_model
import sys

operators= [
    {
        "taskName": "model_1",
        "position": 1,
        "parameters": ""
    },
    {
        "taskName": "model_2",
        "position": 2,
        "parameters": ""
    }
]
def main(userQuery,objectName):
    img_path = objectName
    question = userQuery

    results = {}
    rephrase_prompt= rephrase_model(question)

    # Execute each operator's task
    for operator in operators:
        task_name = operator["taskName"]
        if hasattr(img_m_1, task_name):
            model_func = getattr(img_m_1, task_name)
        elif hasattr(img_m_2, task_name):
            model_func = getattr(img_m_2, task_name)
        else:
            raise ValueError(f"Task {task_name} not found in any module")
        results[task_name] = model_func(img_path, rephrase_prompt)

    final_prompt = f"Given the context derived from analyzing the image, provide a comprehensive answer to the user's question.\n\nContext:\n"
    for key, value in results.items():
        final_prompt += f"{key}: {value}\n\n"
    final_prompt += f"Answer the question: {question}"

    final_response = final_model(
        results.get("model_1"),
        results.get("model_2"),
        final_prompt
    )
    print("Final Answer")
    print(final_response)    
    return {"final_response": final_response}

if __name__ == '__main__':
    userQuery= sys.argv[1]
    objectName=sys.argv[2]

    main(userQuery,objectName)
