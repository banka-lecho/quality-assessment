import warnings

warnings.filterwarnings('ignore')
import torch
from PIL import Image
import clip
import os
from joblib import dump


class ClipModel(object):
    def __init__(self):
        self.arr_embs = []
        self.embeddings = {}
        self.arr_bad_embs = []
        self.bad_embeddings = {}

    def get_embs(self, pathPic=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelCLIP, preprocess = clip.load("ViT-B/32", device=device)
        i = 1
        embeddings = {}
        arr_embs = []
        for filename in os.listdir(pathPic):
            path_file = pathPic + filename
            image = preprocess(Image.open(path_file)).unsqueeze(0).to(device)
            with torch.no_grad():
                one_emb = modelCLIP.encode_image(image)
                embeddings[i] = one_emb
                arr_embs.append(one_emb[0].cpu().numpy())
                i += 1

        return arr_embs, embeddings

    def update_fields(self, path1, path2):
        self.arr_embs, self.embeddings = self.get_embs(path1)
        self.arr_bad_embs, self.bad_embeddings = self.get_embs(path2)


def run_clip(path1, path2):
    my_clip = ClipModel()
    my_clip.update_fields(path1, path2)
    # Сохраняем модель в файл
    model_path = 'saved_models/modelCLIP.joblib'
    dump(my_clip, model_path)
    return my_clip
