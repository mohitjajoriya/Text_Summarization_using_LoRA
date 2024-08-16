
# Text Summarization Model using SAMSum Dataset

This repository contains the implementation of a text summarization model fine-tuned on the SAMSum dataset using Facebook's BART model. The project leverages LoRA, a parameter-efficient fine-tuning (PEFT) method, for efficient model adaptation. The fine-tuned model is deployed on Hugging Face Model Hub for easy access and use.

## Overview
This project focuses on summarizing messenger-like conversations using a fine-tuned version of Facebook's BART model. The SAMSum dataset, which contains 16k conversations and their summaries, is used for training. The LoRA (Low-Rank Adaptation) method is applied for parameter-efficient fine-tuning, allowing the model to adapt efficiently with minimal additional parameters.

## Dataset
- **SAMSum Dataset:** The dataset includes 16,000 conversations with human-written summaries. It is specifically designed for training and evaluating text summarization models on conversational data.
- You can find more details about the dataset on the [SAMSum Dataset page](https://huggingface.co/datasets/samsum).

## Model
- **Base Model:** The base model used is Facebook's BART (Bidirectional and Auto-Regressive Transformers) model, a powerful sequence-to-sequence model pre-trained on a variety of language tasks.
- **LoRA Fine-Tuning:** LoRA is used to fine-tune the model in a parameter-efficient manner, adding minimal parameters while adapting the model to the task.

## LoRA Fine-Tuning
- **LoRA (Low-Rank Adaptation):** A technique for fine-tuning large models with fewer parameters. It introduces rank-decomposition matrices into the layers of a pre-trained model, which are trained while keeping the original weights frozen. This reduces the storage and computational cost of fine-tuning large models.

## Deployment
- The fine-tuned LoRA adapter is deployed on Hugging Face Model Hub, allowing users to easily access and use the model for text summarization tasks.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SharvariMedhe/text-summarization-samsum-lora.git
   cd text-summarization-samsum-lora
