from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

username = ""      # change it to your HuggingFace username

base_checkpoint = username + '/dialogue_Summary'
peft_model_id = username + '/dialogue_Summary_peft'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

# Load Base model
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(base_checkpoint)

# Load PEFT model
loaded_peft_model = PeftModel.from_pretrained(model = peft_model_base,           # The model to be adapted
                                              model_id = peft_model_id,          # Name of the PEFT configuration to use
                                              is_trainable=False,                # False for inference
                                              )



def generate_summary(input, llm):
    """Prepare prompt  -->  tokenize -->  generate output using LLM  -->  detokenize output"""

    input_prompt = f"""
                    Summarize the following conversation.

                    {input}

                    Summary:
                    """

    input_ids = tokenizer(input_prompt, return_tensors='pt')
    tokenized_output = llm.generate(input_ids=input_ids['input_ids'], min_length=30, max_length=200, )
    output = tokenizer.decode(tokenized_output[0], skip_special_tokens=True)

    return output


sample = dataset['test'][0]['dialogue']
label = dataset['test'][0]['summary']

output = generate_summary(sample, llm=loaded_peft_model)

print("Sample")
print(sample)
print("-------------------")
print("Summary:")
print(output)
print("Ground Truth Summary:")
print(label)