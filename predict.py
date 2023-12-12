import os
from typing import List, Text
from cog import BasePredictor, Input, Path

from PIL import Image
import torch
import requests
from transformers import AutoModelForCausalLM, LlamaTokenizer

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()

        print("CUDA Enabled", torch.cuda.is_available())

    def predict(
            self, 
            image: Path = Input(description="input image"), 
            prompt: str = Input(description="input prompt")
        ) -> List[Path]:

        # read image
        image = Image.open(image).convert("RGB")
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            return Text(str(self.tokenizer.decode(outputs[0])).strip("</s>"))

# cog predict -i image=@image.jpg -i prompt="what is the person wearing?"