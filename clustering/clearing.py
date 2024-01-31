import shutil
import os
import copy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# import cairosvg

# похуй пока на эту функцию
def convert_to_png(path_png, path_svg):
    i = 1
    for filename in os.listdir(path_svg):
        svg_file = path_svg + filename
        png_file = path_png + str(i) + '.png'
        extension = os.path.splitext(filename)[1]
        if extension == ".svg":
            break
            # cairosvg.svg2png(url=svg_file, write_to=png_file)
        else:
            source_path = os.path.join(path_svg, filename)
            target_path = os.path.join(path_png, filename)
            shutil.move(source_path, target_path)
        i += 1


def remove_elements(arr, indices):
    new_arr = [x for i, x in enumerate(arr) if i not in indices]
    return new_arr


# сюда я должна подавать изначальные эмбеддинги
def drop_duplicates(arr_all_embs, image_paths):
    all = copy.deepcopy(arr_all_embs)
    similarity = cosine_similarity(all, all)
    columns_to_drop = []
    shape_matrix = similarity.shape[0]
    for i in range(0, shape_matrix):
        for j in range(i + 1, shape_matrix):
            if similarity[i][j] > 0.999999999999999:
                columns_to_drop.append(j)

    columns_to_drop = np.unique(columns_to_drop)

    image_paths = remove_elements(image_paths, columns_to_drop)
    result = remove_elements(arr_all_embs, columns_to_drop)
    return image_paths, result


def drop_unsuitable_pics(arr_all_embs, arr_bad_embs, image_paths):
    # так, тут всё норм с подачей нужных эмбеддингов
    embs = arr_all_embs
    bad_embs = arr_bad_embs
    similarity = cosine_similarity(embs, bad_embs)

    threshold = 0.96
    columns_to_drop = []
    size_all_pics = similarity.shape[0]
    size_bad_pics = similarity.shape[1]
    for i in range(0, size_all_pics):
        for j in range(0, size_bad_pics):
            if similarity[i][j] > threshold:
                columns_to_drop.append(i)

    image_paths = remove_elements(image_paths, columns_to_drop)
    result = remove_elements(arr_all_embs, columns_to_drop)
    return image_paths, result
