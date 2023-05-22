from math import degrees
import sys
import numpy as np
import cv2
import pydiffvg
import torch

from my_shape import PolygonRect, RotationalShapeGroup
sys.path.append("../replace/")
from replaceTile import prepare_model, read, paint
sys.path.append("../retrieve/")
from retriever import retrieve_API, load_images, train_model


if __name__ == "__main__":

    model, images = prepare_model("../retrieve/model.pkl", "../retrieve/dataset_demo")
    PATHPKL = "../demo/results/pkl/"
    tiles = read(PATHPKL + "shapes.pkl", PATHPKL + "shape_groups.pkl")
    canvas = paint(tiles, model, images, canvas_size = (224, 224, 3), name = "result.png")
