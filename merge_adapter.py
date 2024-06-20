import peft as pefty
import argparse
import os
import json
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
import sys

#Create the parser
parser = argparse.ArgumentParser(description="--input_dir --model_name")

# Add arguments
parser.add_argument('--output_dir', type=str)
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('--model_adapter',type=str)
parser.add_argument('--revision',type=str)

args = parser.parse_args()
model_name = args.model_name

base_model_name = model_name
adapter_filename = args.model_adapter

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    #device_map='auto',
)
base_model.config.use_cache = False
print(base_model)

footprint = base_model.get_memory_footprint()
print("BASE MEM FOOTPRINT",footprint)

#base_model.load_adapter(adapter_filename)
#footprint = base_model.get_memory_footprint()
#print("ADAPTER MEM FOOTPRINT",footprint)

model_to_merge = pefty.PeftModel.from_pretrained(base_model,adapter_filename)
print("MODEL MERGED")

merged_model = model_to_merge.merge_and_unload()
merged_model.push_to_hub(args.output_dir)
print("Model saved")

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.push_to_hub(args.output_dir)
print("Tokenizer saved")
