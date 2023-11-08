from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_name = "Alpi157/Final_conversational_model"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_response(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,
                            pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)


    match = re.search(r'вопрос:.*?ответ:(.*?)вопрос:', response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer

    return None


# Example prompt
prompt = "вопрос: Какая студенческая жизнь в вашем универе?"
response = generate_response(prompt)
if response is not None:
    print(response)
