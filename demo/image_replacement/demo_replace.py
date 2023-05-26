from math import degrees
import sys
import numpy as np
import cv2
import pydiffvg
import torch
import argparse

from my_shape import PolygonRect, RotationalShapeGroup
sys.path.append("../replace/")
from replaceTile import prepare_model, read, paint
sys.path.append("../retrieve/")
from retriever import retrieve_API, load_images, train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model", default="../retrieve/model.pkl")
    parser.add_argument("--dataset", help="path to dataset", default="../retrieve/dataset_demo")
    parser.add_argument("--shapes", help="path to shapes.pkl and shape_groups.pkl", default="../demo/results/pkl/")
    parser.add_argument("--output", help="name of output image", default="result")
    args = parser.parse_args()
    
    # model, images = prepare_model("../retrieve/model.pkl", "../retrieve/dataset_demo")
    model, images = prepare_model(args.model, args.dataset)
    # PATHPKL = "../demo/results/pkl/"
    PATHPKL = args.shapes
    tiles = read(PATHPKL + "shapes.pkl", PATHPKL + "shape_groups.pkl")
    # canvas = paint(tiles, model, images, canvas_size = (224, 224, 3), name = "result.png")
    outname = args.output
    if not (outname.endswith(".png") or outname.endswith(".jpg")):
        outname += ".png"
    canvas = paint(tiles, model, images, canvas_size = (224, 224, 3), name = outname)
