import openai
from colorama import init, Fore
import time


def generate_response_gpt(prompt):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )

    return response.choices[0].message.content

def generate_response_lama_server(prompt):

    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="tom",
    )

    response_text = ""
    response = client.chat.completions.create(
        model="llama.cpp/models/mistral-7b-instruct-v0.1.Q4_0.gguf",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
        max_tokens=1000,
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
    return response_text


def generate_response(prompt, model):
    if model == "gpt":
        return generate_response_gpt(prompt)
    elif model == "qwen":
        return generate_response_lama_server(prompt)
    else:
        return "Model not found"



def create_tokenized_prompt(prompt_text, tokenizer, device):
    """Tokenize and prepare the prompt for the model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer([text], return_tensors="pt").to(device)

def generate_response_local_model(model, model_inputs):
    """Generate a response using the model."""
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return generated_ids



if __name__ == "__main__":

    prompts = [
        "what is ROI in the context of finance, provide a worked example? (be concise)",
        "define the efficient frontier in the context of finance (be concise)"
    ]

    for prompt in prompts:
        print(Fore.LIGHTMAGENTA_EX + prompt, end="\n")
        answer = generate_response_lama_server(prompt)
        print(Fore.LIGHTBLUE_EX + answer, end="\n\n")
