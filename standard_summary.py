from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import clean_text
from constants import model_name, device, standard_instruct, standard_summary, text, test_text

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_summary(original_text: str) -> str:

  messages = [
      {"role": "user", "content": standard_instruct + text},
      {"role": "assistant", "content": standard_summary},
      {"role": "user", "content": standard_instruct + original_text}
  ]

  encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
  model_inputs = encodeds.to(device)

  model.to(device)

  generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
  decoded = tokenizer.batch_decode(generated_ids)

  return clean_text(decoded[0])


summary = generate_summary(test_text)
print(summary)