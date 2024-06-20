from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
import sys
import argparse

#Create the parser
parser = argparse.ArgumentParser(description="--input_dir --model_name")

# Add arguments
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")

args = parser.parse_args()
model_name = args.model_name

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
)
