from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import dump
from visualization import visualize_dependency


def get_eps(embs):
    matrix = cosine_similarity(embs, embs)
    k = 4
    nbrs = NearestNeighbors(n_neighbors=k).fit(matrix)
    distances, indices = nbrs.kneighbors(matrix)
    k_distances = distances[:, -1]
    k_distances.sort()

    plt.plot(list(range(1, len(matrix) + 1)), k_distances)
    plt.xlabel('Номер точки')
    plt.ylabel(f'{k}-ое расстояние')
    plt.savefig('visualize_result/dbscan/graphic_of_eps.png')


class ModelDBSCAN(object):

    def __init__(self, embs, matrix_similar):
        self.embs = embs
        self.matrix_similar = matrix_similar
        self.best_model = DBSCAN()
        self.best_score = 0

    def dbscan_result(self, distance_metric):
        eps_list = np.arange(start=3, stop=3.5, step=0.01)
        min_sample_list = np.arange(start=2, stop=20, step=3)
        max_sil_score = -1
        best_model = DBSCAN()
        for eps_trial in tqdm(eps_list):
            for min_sample_trial in min_sample_list:
                db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial, metric=distance_metric)
                result = db.fit_predict(self.matrix_similar)
                try:
                    sil_score = silhouette_score(self.matrix_similar, result)
                except ValueError:
                    sil_score = 0
                if sil_score > max_sil_score:
                    max_sil_score = sil_score
                    self.best_model = db

        return best_model, max_sil_score


def run_dbscan(embs, similarity):
    my_metrics = ['euclidean', 'l2', 'canberra', 'hamming']
    max_sil_score = -1
    modelDBSCAN = ModelDBSCAN(embs, similarity)
    for i in tqdm(my_metrics):
        model, sil_score = modelDBSCAN.dbscan_result(i)
        if max_sil_score < sil_score:
            max_sil_score = sil_score
            modelDBSCAN.best_model = model

    modelDBSCAN.best_score = max_sil_score
    model_path = 'saved_models/modelDBSCAN.joblib'
    dump(modelDBSCAN.best_model, model_path)

    return modelDBSCAN.best_model


class ModelKMEANS(object):

    def __init__(self, embs, matrix_similar):
        self.embs = embs
        self.matrix_similar = matrix_similar
        self.best_model = KMeans()
        self.best_score = 0

    def kmeans_result(self):
        count_clusters_list = range(250, 265)
        max_sil_score = -1
        scores = []
        for count_clusters in tqdm(count_clusters_list):
            kmeans = KMeans(n_clusters=count_clusters)
            result = kmeans.fit_predict(self.matrix_similar)
            sil_score = silhouette_score(self.matrix_similar, result)
            scores.append(sil_score)
            if max_sil_score < sil_score:
                max_sil_score = sil_score
                self.best_model = kmeans

        self.best_score = max_sil_score
        visualize_dependency(scores, count_clusters_list, 'visualize_result/dbscan/scores_clusters.png')
        return self.best_model.labels_, max_sil_score


def run_kmeans(embs, similarity):
    model_kmeans = ModelKMEANS(embs, similarity)
    model_kmeans.kmeans_result()
    model_path = 'saved_models/modelKMEANS.joblib'
    dump(model_kmeans.best_model, model_path)

    return model_kmeans.best_model
