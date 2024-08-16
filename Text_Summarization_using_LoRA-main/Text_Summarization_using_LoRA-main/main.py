import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model, TaskType

import warnings
warnings.filterwarnings('ignore')

# Load model from HF Model Hub

"""
BART HAS 400M PARAMS: https://github.com/facebookresearch/fairseq/tree/main/examples/bart
Look into Model card - 400 Million parameters
"""

checkpoint = "facebook/bart-large-cnn"                # username/repo-name

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Load SAMSum dataset
dataset = load_dataset("samsum")

# Define function to prepare dataset

def tokenize_inputs(example):

    start_prompt = "Summarize the following conversation.\n\n"
    end_prompt = "\n\nSummary: "
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, return_tensors='pt').input_ids             # 'pt' for pytorch tensor
    example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True, return_tensors='pt').input_ids

    return example


# Prepare dataset
tokenizer.pad_token = tokenizer.eos_token
tokenized_datasets = dataset.map(tokenize_inputs, batched=True)       # using batched=True for Fast tokenizer implementation

# Remove columns/keys that are not needed further
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'dialogue', 'summary'])

# Shortening the data: Just picking row index divisible by 100
# For learning purpose! It will reduce the compute resource requirement and training time

tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)



# LoRA-specific parameters
lora_config = LoraConfig(
    r=32,                       # 8, 16, 32    # the rank of the update matrices
    lora_alpha=32,                             # LoRA scaling factor
    lora_dropout=0.05,
    bias='none',                               # specifies if the bias parameters should be trained
    task_type=TaskType.SEQ_2_SEQ_LM,           # telling lora that this is a sq2seq modeling task
)

# Trainable PEFTModel
peft_model = get_peft_model(model, peft_config=lora_config)



peft_training_args = TrainingArguments(
    output_dir="./mode_tuned_peft",           # local directory
    learning_rate=1e-5,
    num_train_epochs=5,      ## for 5 epochs took around 10 minutes
    weight_decay=0.01,
    auto_find_batch_size=True,
    evaluation_strategy='epoch',
    logging_steps=10
)

peft_trainer = Trainer(
    model=peft_model,                    # model to be fine-tuned
    args=peft_training_args,                       # training arguments
    train_dataset=tokenized_datasets['train'],          # train data to use
    eval_dataset=tokenized_datasets['validation']       # validation data to use
)

# Training
peft_trainer.train()

# !huggingface-cli login


# Push peft adapter to Hub

my_peft_repo = "dialogue_Summary_peft"

peft_model.push_to_hub(repo_id= my_peft_repo, commit_message= "Upload peft adapter", )



