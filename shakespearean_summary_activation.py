from typing import Dict, Union, List

import torch
from transformers import AutoTokenizer

from transformer_lens import HookedTransformer

from constants import model_name, device, standard_instruct, standard_summary, text, test_text

torch.set_grad_enabled(False)  # save memory
model = HookedTransformer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
model.to(device)

SEED = 0
sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)

# steering prompts
prompt_add, prompt_sub = "Write in Shakespearean English", "Do not write in Shakespearean English"
coeff = 5
act_name = 6
# prompt = "I went up to my friend and said"
prompt = test_text + "\nSummarize in Shakespearean English:\n"

#Padding
tlen = lambda prompt: model.to_tokens(prompt).shape[1]
pad_right = lambda prompt, length: prompt + model.tokenizer.eos_token * (length - tlen(prompt))
l = max(tlen(prompt_add), tlen(prompt_sub))
prompt_add, prompt_sub = pad_right(prompt_add, l), pad_right(prompt_sub, l)

print(f"'{prompt_add}'", f"'{prompt_sub}'")

""" ## Get activations"""

def get_resid_pre(prompt: str, layer: int):
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]


act_add = get_resid_pre(prompt_add, act_name)
act_sub = get_resid_pre(prompt_sub, act_name)
act_diff = act_add - act_sub
print(act_diff.shape)

""" ## Generate from the modified and non-modified model"""

def ave_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return  # caching in model.generate for new tokens

    # We only add to the prompt (first call), not the generated tokens.
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

    # add to the beginning (position-wise) of the activations
    resid_pre[:, :apos, :] += coeff * act_diff


def hooked_generate(prompt_batch: List[str], fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)
    
    tokenized = model.to_tokens(prompt_batch)

    with model.hooks(fwd_hooks=fwd_hooks):
        steered_r = model.generate(input=tokenized, max_new_tokens=100, do_sample=True, **kwargs)
    
    basic_r = model.generate(input=tokenized, max_new_tokens=100, do_sample=True, **kwargs)

    return steered_r, basic_r


editing_hooks = [(f"blocks.{act_name}.hook_resid_pre", ave_hook)]
steered_res, basic_res = hooked_generate(prompt, editing_hooks, seed=SEED, **sampling_kwargs)

# Print results
steered_res_str = model.to_string(steered_res[:, 1:])
print(steered_res_str)

basic_res_str = model.to_string(basic_res[:, 1:])
print(basic_res_str)