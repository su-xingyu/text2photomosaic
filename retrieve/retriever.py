from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, BallTree, KDTree
from sklearn.svm import SVC
import pickle
import colorsys

def dominant_color(image, k=3, n_init=10):
    image_arr = np.array(image.convert('RGB')).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=n_init).fit(image_arr)
    # print (kmeans.cluster_centers_[0])
    return kmeans.cluster_centers_[0]

def load_images(image_folder='/content/images'):
    return [{"image": Image.open(f), "filename": f.name} for f in Path(image_folder).iterdir() if f.is_file()]

def train_model(images, algorithm='plain', size_weight=0.1):
    if algorithm == 'plain':
        return None

    features = [np.hstack([dominant_color(img['image']), img['image'].size[0] * size_weight, img['image'].size[1] * size_weight]) for img in images]
    features = np.array(features)

    if algorithm == 'knn':
        model = KNeighborsClassifier(n_neighbors=1).fit(features, range(len(images)))
    elif algorithm in ('balltree', 'kdtree'):
        model = {'balltree': BallTree, 'kdtree': KDTree}[algorithm](features)
    elif algorithm == 'svm':
        model = SVC(kernel='linear', C=1).fit(features, range(len(images)))
    else:
        raise ValueError("Invalid algorithm specified. Use 'plain', 'knn', 'balltree', 'kdtree', or 'svm'.")

    return model

def query_model(model, images, target_color, target_size, algorithm='plain', size_weight=0):
    if algorithm == 'plain':
        size_diffs = [np.linalg.norm(target_color - dominant_color(img['image'])) + size_weight * (abs(target_size[0] - img['image'].size[0]) + abs(target_size[1] - img['image'].size[1])) for img in images]
        closest_img = images[np.argmin(size_diffs)]
        return closest_img["image"]

    query_point = np.hstack([target_color, target_size[0] * size_weight, target_size[1] * size_weight])

    if algorithm in ('knn', 'svm'):
        index = model.predict([query_point])[0]
    elif algorithm in ('balltree', 'kdtree'):
        index = model.query([query_point], return_distance=False)[0][0]
    else:
        raise ValueError("Invalid algorithm specified. Use 'plain', 'knn', 'balltree', 'kdtree', or 'svm'.")

    return images[index]["image"]

def retrieve_API(target_color, target_size, model, images, algorithm='plain'):
    return query_model(model, images, target_color, target_size, algorithm=algorithm)

if __name__ == "__main__":
    image_folder = './dataset_demo'
    algorithm = 'kdtree'

    images = load_images(image_folder)
    model = train_model(images, algorithm=algorithm)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    query_colors = [np.array([100, 150, 200]), np.array([255, 20, 10]), np.array([10, 255, 20])]
    query_sizes = [(600, 400), (800, 600), (1024, 768)]

    for query_color, query_size in zip(query_colors, query_sizes):
        retrieved_image = retrieve_API(query_color, query_size, loaded_model, images, algorithm=algorithm)
        print(f"Closest image (Color: {query_color}, Size: {query_size}): {retrieved_image.filename}")

        retrieved_image.show()