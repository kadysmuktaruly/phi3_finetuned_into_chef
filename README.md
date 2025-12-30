# anyone could run this and use the finetuned model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

base = "microsoft/Phi-3-mini-4k-instruct"
path = "phi3-recipes-lora"  # your saved folder

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    path,
    quantization_config=bnb,
    device_map="auto",
)
model.eval()
