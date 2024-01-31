import numpy as np
from torch import nn
import os
import glob
from torch.nn.functional import pdist
from sklearn.metrics import silhouette_score
from blipModel import run_blip
import zipfile
from clearing import drop_duplicates, drop_unsuitable_pics
from clustering_models import get_eps, run_dbscan, run_kmeans
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load

from visualization import visualize, pca_visualization

warnings.filterwarnings('ignore')
from clipModel import run_clip


def get_similarities(embs, pdist=False):
    matrix_similar = []
    for index_target in range(1, file_count + 1):
        similarities = []
        target_embedding = embs[index_target]
        for i in range(1, len(embs) + 1):
            embedding = embs[i]
            if (pdist):
                pdist = nn.PairwiseDistance(p=2)
                output = pdist(target_embedding, embedding)
            else:
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                output = cos(target_embedding, embedding)
            similarities.append(output.item())

        matrix_similar.append(similarities)
    matrix_similar = np.array(matrix_similar)
    return matrix_similar


def dbscan(arr_embs, similarity):
    # check eps for DBSCAN
    get_eps(arr_embs)
    file.write("run DBSCAN")
    file.write("\n")
    dbscan_model_path = 'saved_models/modelDBSCAN.joblib'
    dbscan_model = load(dbscan_model_path) if os.path.exists(dbscan_model_path) else run_dbscan(arr_embs,
                                                                                                similarity)
    try:
        dbscan_sil_score = silhouette_score(similarity, dbscan_model.labels_)
    except ValueError:
        dbscan_sil_score = 0

    file.write("Model DBSCAN")
    file.write("\n")
    file.write("Silhouette score: " + str(dbscan_sil_score))
    file.write("\n")
    visualize(dbscan_model.labels_, similarity, 'visualize_result/dbscan/distribution.png')
    pca_visualization(similarity, dbscan_model.labels_, 'visualize_result/dbscan/pca.png')
    return dbscan_sil_score, dbscan_model.labels_


def kmeans(arr_embs, similarity):
    file.write("run K-Means")
    file.write("\n")
    kmeans_model_path = 'saved_models/modelKMEANS.joblib'
    kmeans_model = load(kmeans_model_path) if os.path.exists(kmeans_model_path) else run_kmeans(arr_embs,
                                                                                                similarity)
    kmeans_sil_score = silhouette_score(similarity, kmeans_model.labels_)

    file.write("Model K-Means: ")
    file.write("\n")
    file.write("Silhouette score: " + str(kmeans_sil_score))
    file.write("\n")
    visualize(kmeans_model.labels_, similarity, 'visualize_result/kmeans/distribution.png')
    pca_visualization(similarity, kmeans_model.labels_, 'visualize_result/kmeans/pca.png')
    return kmeans_sil_score, kmeans_model.labels_


def zip_folder(folder_path, output_name):
    with zipfile.ZipFile(output_name, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".npz"):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))


def check_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Директория {path} создана')
    else:
        print(f'Директория {path} уже существует')


if __name__ == '__main__':
    path = 'data/pictures/'
    path_bad_pics = 'data/badPictures/'
    os.listdir(path)
    file_count = len(glob.glob(path + '/*'))
    image_paths = [os.path.join(path, file) for file in os.listdir(path) if
                   file.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    path_models = 'saved_models'
    check_directory(path_models)
    path_result = 'result'
    check_directory(path_result)

    with open('output.txt', 'w') as file:
        # got results of clip model
        file.write("Getting embs")
        file.write("\n")
        clip_model_path = 'saved_models/modelCLIP.joblib'
        clip_model = load(clip_model_path) if os.path.exists(clip_model_path) else run_clip(path, path_bad_pics)

        # got results of blip model
        # file.write("Getting texts")
        # blip_model_path = 'saved_models/modelBLIP.joblib'
        # blip_model = load(blip_model_path) if os.path.exists(blip_model_path) else run_blip('data/pictures/')
        # texts = blip_model.texts
        # np.savez('result/texts.npz', *texts)

        arr_embs = clip_model.arr_embs
        dict_embs = clip_model.embeddings
        arr_bad_embs = clip_model.arr_bad_embs
        dict_bad_embs = clip_model.bad_embeddings

        file.write("Cleaning data")
        file.write("\n")
        image_paths, arr_embs = drop_duplicates(arr_embs, image_paths)
        image_paths, arr_embs = drop_unsuitable_pics(arr_embs, arr_bad_embs, image_paths)

        # create cosine matrix similar
        similarity = cosine_similarity(arr_embs, arr_embs)

        dbscan_score, dbscan_labels = dbscan(arr_embs, similarity)
        kmeans_score, kmeans_labels = kmeans(arr_embs, similarity)

        labels = dbscan_labels if (dbscan_score > kmeans_score) else kmeans_labels

        np.savez('result/labels.npz', *labels)
        np.savez('result/embeddings.npz', *arr_embs)

        zip_folder('result', 'result.zip')
