from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def init_llama(model_id="meta-llama/Meta-Llama-3-8B-Instruct",
               use_device_map_auto=True,
               dtype_if_cuda=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if use_device_map_auto:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype_if_cuda
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_if_cuda
        )
        mdl.to("cuda")
    return tok, mdl

@torch.inference_mode()
def get_response(prompt, tm, mm, max_tokens=200, use_chat_template=True):
    if use_chat_template and hasattr(tm, "apply_chat_template"):
        chat = [
            {"role": "system", "content": "You are a concise driving safety classifier. Respond exactly as instructed."},
            {"role": "user",   "content": prompt}
        ]
        input_text = tm.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tm(input_text, return_tensors="pt")
    else:
        inputs = tm(prompt, return_tensors="pt")

    out = mm.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tm.eos_token_id
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tm.decode(gen_ids, skip_special_tokens=True).strip()