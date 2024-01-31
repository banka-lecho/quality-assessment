from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import os


def visualize(labels, matrix_similar, path):
    plt.scatter(matrix_similar[:, 0], matrix_similar[:, 1], c=labels, cmap='viridis')
    plt.title('Distribution')
    plt.savefig(path)


def visualize_images(labels):
    i = 0
    fig, ax = plt.subplots()
    path = '/Users/anastasiaspileva/Desktop/metric/visualize_result'
    for filename in os.listdir(path):
        url = path + filename
        img_class = labels[i]
        img = Image.open(url)
        ax.imshow(img, extent=[0, 10, 0, 10], alpha=0.5)
        ax.annotate(img_class, xy=(5, 5), xytext=(7, 7), arrowprops=dict(facecolor='black'))
        i += 1

    plt.show()


def pca_visualization(X, y, path):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(14, 12))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.colorbar()
    plt.savefig(path)


def visualize_dependency(arr1, arr2, path):
    plt.plot(arr1, arr2)
    plt.title('Dependency')
    plt.savefig(path)
