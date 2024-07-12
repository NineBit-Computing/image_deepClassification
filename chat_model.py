import ollama

# Language mode llama3 from ollama
def final_model(response1,response2,final_prompt):
    
    final_response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role" : "user",
                "content" : final_prompt,
            }
        ]
    )
    return final_response['message']['content']