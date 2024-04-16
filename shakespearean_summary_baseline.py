
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import model_name, device, shakespearean_instruct, shakespearean_summary, text, test_text
from utils import clean_text

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_summary(original_text: str) -> str:

  messages = [
      {"role": "user", "content": shakespearean_instruct + text},
      {"role": "assistant", "content": shakespearean_summary},
      {"role": "user", "content": shakespearean_instruct + original_text}
  ]

  encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

  model_inputs = encodeds.to(device)
  model.to(device)

  generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)

  decoded = tokenizer.batch_decode(generated_ids)

  return clean_text(decoded[0])


summary = generate_summary(test_text)
print(summary)