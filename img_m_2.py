import ollama

# Mult-model llava from ollama
def model_2(image_path,prompt):
    response2 = ollama.chat(
        model="llava",
        messages=[
            {
                "role" : "user",
                "content" : prompt,
                "images" :[image_path]
            }
        ]
    )
    print("Model_2 Response :", response2['message']['content'])
    return response2
