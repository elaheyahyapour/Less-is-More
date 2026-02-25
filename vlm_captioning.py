from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

def _first_param_device(module):
    for p in module.parameters():
        return p.device
    return torch.device("cpu")

def _get_vision_device(model):

    vm = None
    if hasattr(model, "vision_model") and model.vision_model is not None:
        vm = model.vision_model
    else:
        try:
            vm = model.get_submodule("vision_model")
        except Exception:
            vm = None
    if vm is None:
        return _first_param_device(model)
    return _first_param_device(vm)

def init_blip2(device="cuda", use_device_map_auto=True, dtype_if_cuda=torch.float16):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    if use_device_map_auto:
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            device_map="auto",
            torch_dtype=(dtype_if_cuda if device.startswith("cuda") else torch.float32),
        )
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=(dtype_if_cuda if device.startswith("cuda") else torch.float32),
        )
        model.to(device)

    return processor, model

@torch.inference_mode()
def generate_caption(image_path, processor, model, prefer_half_on_cuda=True):

    image = Image.open(image_path).convert("RGB")
    prompt = ("Describe the driving scene focusing on dynamic agents, "
              "distances, traffic controls, occlusions, and any potential hazards.")
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    devices = set()
    for n, p in model.named_parameters():
        if p.device is not None:
            devices.add(p.device)
            if len(devices) > 1:
                break

    if len(devices) == 1:
        dev = list(devices)[0]
        target_dtype = torch.float16 if (dev.type == "cuda" and prefer_half_on_cuda) else torch.float32
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(dev, dtype=target_dtype if k == "pixel_values" else None)
    else:
        vision_dev = _get_vision_device(model)
        target_dtype = torch.float16 if (vision_dev.type == "cuda" and prefer_half_on_cuda) else torch.float32
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(vision_dev, dtype=target_dtype)

    output = model.generate(**inputs, max_new_tokens=60)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption
