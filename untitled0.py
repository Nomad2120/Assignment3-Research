# -*- coding: utf-8 -*-
"""Untitled0.ipynb


!pip install transformers

! pip install -U accelerate
! pip install -U transformers

import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def load_dataset(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    qa_pairs = []
    current_question = None

    for line in lines:
        line = line.strip()
        if line.startswith("вопрос: "):
            current_question = line.replace("вопрос: ", "")
        elif line.startswith("ответ: "):
            answer = line.replace("ответ: ", "")
            if current_question is not None:
                qa_pairs.append((current_question, answer))
                current_question = None

    return qa_pairs

train_file_path = "tbot_dataset.txt"  # path to dataset file
output_dir = "chatbot_model"  # directory where the model will be saved
model_name = "ai-forever/rugpt3medium_based_on_gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

qa_pairs = load_dataset(train_file_path, tokenizer)
input_texts = [pair[0] for pair in qa_pairs]
target_texts = [pair[1] for pair in qa_pairs]

input_ids = [tokenizer.encode(text, return_tensors="pt") for text in input_texts]
target_ids = [tokenizer.encode(text, return_tensors="pt") for text in target_texts]

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file_path,
    block_size=128,  # Adjust the block size as needed
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=6,
    save_steps=500,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

def generate_text(prompt, model, tokenizer, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        top_k=50,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # check if generated text contains a valid answer (not a new question)
    if "ответ:" in generated_text.lower():
        return generated_text
    else:
        # handle cases where it generated a new question
        return "I'm sorry, I couldn't provide a valid answer."

prompt = "вопрос: Как поступить в аиту?"
response = generate_text(prompt, model, tokenizer)
print(response)

model_path = "/content/chatbot_model"  # path to model directory
output_drive_path = "/content/drive/My Drive/GPT"  # path in Google Drive

import shutil

shutil.copytree(model_path, output_drive_path)

import os
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

!huggingface-cli login

from transformers import GPT2LMHeadModel

#path to locally trained model
local_model_path = "/content/chatbot_model"

# loading the locally trained model
model = GPT2LMHeadModel.from_pretrained(local_model_path)

# a name for model on the Hugging Face Model Hub
model_name = "Alpi157/Final_conversational_model"  # replace "your_username" with your Hugging Face username

# pushing the model to the Hugging Face Model Hub
model.push_to_hub(model_name)

print(f"Model '{model_name}' has been pushed to the Hugging Face Model Hub.")

transformers-cli login

!huggingface-cli repo-metadata update "Alpi157/Final_conversational_model" --type model --task "Conversational"
