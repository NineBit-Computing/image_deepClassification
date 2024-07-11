import ollama
import json

def llm_model(prompt):
    final_response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    print(final_response["message"]["content"])

question = "Is there any hazardous material in this picture?"

prompt = f"""
Rephrase the following question to include more relevant details that provide a comprehensive understanding of the subject. Consider the following aspects:

1. Identify the main subject (person, object, or action).
2. Add details about distinguishing features or characteristics.
3. Include information about related actions or behaviors.
4. Provide context or purpose if relevant.

Return only the rephrased question in inside rephrased_question : . 

**Examples**:

0. Original Question: "{question}"
   Rephrased Question: "{question},Identify what objects are visible in the image?"

1. Original Question: "What kind of bird is shown here?"
   Rephrased Question: "What kind of bird is shown here, what are its distinguishing features, what is its color, and what is the bird doing?"

2. Original Question: "Is the person in the image smiling?"
   Rephrased Question: "Is the person in the image smiling, what is their overall facial expression, what are they wearing, and what are they doing?"

3. Original Question: "What does the text in this image say?"
   Rephrased Question: "What does the text in this image say, what language is it in, what font style is used, and what is the context or purpose of the text?"

4. Original Question: "Is someone cooking in the image?"
   Rephrased Question: "Is someone cooking in the image, what are they cooking, what utensils are they using, and what is the person’s facial expression?"

**Now rephrase this question**: "{question}"
"""

llm_model(prompt)
