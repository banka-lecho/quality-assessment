from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm
from PIL import Image
import os
from joblib import dump


class BlipModel(object):

    def __init__(self, path: str):
        self.path = path
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.modelBLIP = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.texts = []

    def get_texts(self):
        self.texts = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelBLIP.to(device)
        for filename in tqdm(os.listdir(self.path)):
            url = self.path + filename
            image = Image.open(url).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(device, torch.float16)

            generated_ids = self.modelBLIP.generate(**inputs, max_new_tokens=3)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            self.texts.append(generated_text)

        return self.modelBLIP


def run_blip(path):
    model_blip = BlipModel(path)
    model_blip.get_texts()
    model_path = 'saved_models/modelCLIP.joblib'
    dump(model_blip, model_path)
    return model_blip
