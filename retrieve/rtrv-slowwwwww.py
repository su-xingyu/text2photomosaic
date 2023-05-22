import os
import numpy as np
from PIL import Image

def resize(img, size):
    return np.array(Image.fromarray(img).resize(size))

def build_image_pyramid(image, levels=4):
    pyramid = [image]
    for i in range(levels - 1):
        image = resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        pyramid.append(image)
    return pyramid
  
def get_dominant_color(image):
    if len(image.shape) == 2:
        # Grayscale image
        pixel_count = np.zeros(256)
        for pixel_value in image.flatten():
            pixel_count[pixel_value] += 1
        index = np.argmax(pixel_count)
        r, g, b = index, index, index
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        pixel_count = np.zeros((256, 256, 256))
        for r, g, b in image.reshape(-1, 3):
            pixel_count[r, g, b] += 1
        indices = np.unravel_index(np.argmax(pixel_count), pixel_count.shape)
        r, g, b = indices[0], indices[1], indices[2]
    else:
        raise ValueError("Unsupported image format. Expected RGB or grayscale image.")
    
    return np.array([r, g, b])



def retrieve_closest_image(target_color, target_size, image_dir):
    min_distance = float('inf')
    closest_image_path = None
    
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            file_path = os.path.join(image_dir, file_name)
            try:
                img = np.array(Image.open(file_path))
            except:
                continue
            
            img_size = (img.shape[1], img.shape[0])
            pyramid = build_image_pyramid(img)
            
            for i, level in enumerate(pyramid[::-1]):
                level_size = (level.shape[1], level.shape[0])
                if i == 0:
                    distance_weight = 1.0
                else:
                    distance_weight = 0.1
                size_weight = ((level_size[0] / target_size[0]) + (level_size[1] / target_size[1])) * 0.1
                
                dominant_color = get_dominant_color(level)
                distance = np.linalg.norm(dominant_color - target_color)
                total_distance = distance * distance_weight + size_weight
                
                if total_distance < min_distance:
                    min_distance = total_distance
                    closest_image_path = file_path
    
    return closest_image_path

target_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # sequence of RGB colors
target_size = (100, 100)  # Example target size
image_dir = '/content/images'

for color in target_colors:
    closest_image_path = retrieve_closest_image(color, target_size, image_dir)
    
    if closest_image_path is not None:
        closest_image = Image.open(closest_image_path)
        closest_image.show() 
    else:
        print("No closest image found for the color:", color)

from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, BallTree, KDTree
from sklearn.svm import SVC
import pickle

def dominant_color(image, k=3, n_init=10, color_space='RGB'):
    if color_space.upper() == 'HSV':
        image = image.convert('HSV')
    elif color_space.upper() != 'RGB':
        raise ValueError("Invalid color space specified. Use 'RGB' or 'HSV'.")
    image_arr = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=n_init).fit(image_arr)
    return kmeans.cluster_centers_[0]

def load_images(image_folder='/content/images'):
    return [{"image": Image.open(f), "filename": f.name} for f in Path(image_folder).iterdir() if f.is_file()]

def train_model(images, algorithm='plain', size_weight=0.1, color_space='RGB'):
    if algorithm == 'plain':
        return None

    features = [np.hstack([dominant_color(img['image'], color_space=color_space), img['image'].size[0] * size_weight, img['image'].size[1] * size_weight]) for img in images]
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

def query_model(model, images, target_color, target_size, algorithm='plain', size_weight=0.1, color_space='RGB'):
    if algorithm == 'plain':
        size_diffs = [np.linalg.norm(target_color - dominant_color(img['image'], color_space=color_space)) + size_weight * (abs(target_size[0] - img['image'].size[0]) + abs(target_size[1] - img['image'].size[1])) for img in images]
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

def retrieve_API(target_color, target_size, model, images, algorithm='plain', color_space='RGB'):
    return query_model(model, images, target_color, target_size, algorithm=algorithm, color_space=color_space)

if __name__ == "__main__":
    image_folder = '/content/images'
    algorithm = 'svm'
    color_space = 'RGB' #HSV sometimes works not so well

    images = load_images(image_folder)
    model = train_model(images, algorithm=algorithm, color_space=color_space)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    query_colors = [np.array([100, 150, 200]), np.array([255, 20, 10]), np.array([10, 255, 20])]
    query_sizes = [(600, 400), (800, 600), (1024, 768)]

    for query_color, query_size in zip(query_colors, query_sizes):
        retrieved_image = retrieve_API(query_color, query_size, loaded_model, images, algorithm=algorithm, color_space=color_space)
        print(f"Closest image (Color: {query_color}, Size: {query_size}): {retrieved_image.filename}")
        retrieved_image.show()

import numpy as np
from PIL import Image
import os

def generate_color_image(color, size):
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:, :] = color
    return Image.fromarray(image)

target_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
target_size = (100, 100)
image_dir = '/content/images'

# Create the directory if it doesn't exist
os.makedirs(image_dir, exist_ok=True)

for i, color in enumerate(target_colors):
    image = generate_color_image(color, target_size)
    image_path = os.path.join(image_dir, f"{['red', 'green', 'blue'][i]}.png")
    image.save(image_path)

import numpy as np
from PIL import Image
import os

def generate_color_image(color, size):
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:, :] = color
    return Image.fromarray(image)

target_size = (100, 100)
image_dir = '/content/images'

# Create the directory if it doesn't exist
os.makedirs(image_dir, exist_ok=True)

for i in range(100):
    # Generate a random RGB color
    color = np.random.randint(0, 256, size=3)
    image = generate_color_image(color, target_size)
    image_path = os.path.join(image_dir, f"{i}.png")
    image.save(image_path)

print("Images saved successfully.")