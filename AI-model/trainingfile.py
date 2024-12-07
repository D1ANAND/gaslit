# -*- coding: utf-8 -*-

# Importing the necessary dependencies
import torch
# checking the device capabilities
major_version, minor_version = torch.cuda.get_device_capability()
# importing unsloth from github and installing
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes

!pip install triton

!pip uninstall xformers -y
!pip install xformers==0.0.27

# Importing the model
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Loading the llama3 model as the inference model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# The datasets library that will import our training data
!pip install datasets

# Importing the Supervised Fintuning Trainer
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
max_seq_length = 2048

# url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
# # Loading the dataset from the hugging face repository
# dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

dataset = load_dataset("DEEPAK70681/CryptoSendIntentv2", split = "train")

print(dataset[:5])

def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# 4. Do model patching and add fast LoRA weights and training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # Rank stabilized LoRA
    loftq_config = None, # LoftQ
)
# 5. using the supervised fine tuning trainer to fine tune the model
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size=8,  # Increase batch size if memory permits
        gradient_accumulation_steps=2,  # Adjust based on batch size
        warmup_steps=100,               # Increased warmup steps
        max_steps=500,                  # Increased steps
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,               # Adjusted logging frequency
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,

    ),
)
trainer.train()



model.save_pretrained("Inference-Engine")


model.push_to_hub("Inference-Engine")
