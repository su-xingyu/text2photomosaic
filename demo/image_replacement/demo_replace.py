from math import degrees
import sys
import argparse
import os

sys.path.append("../mosaic_generation/")
from my_shape import PolygonRect, RotationalShapeGroup
from replaceTile import prepare_model, read, paint
from retrieve.retriever import retrieve_API, load_images, train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model", default="./model.pkl")
    parser.add_argument("--dataset", help="path to dataset", default="retrieve/dataset_demo")
    parser.add_argument("--shapes", help="path to shapes.pkl", 
                        default="../results/previous_results/clip/exp1/pkls/clip_shapes.pkl")
    parser.add_argument("--shapes_groups", help="path to shape_groups.pkl", 
                        default="../results/previous_results/clip/exp1/pkls/clip_shape_groups.pkl")
    parser.add_argument("--output", help="name of output image", default="result.png")
    args = parser.parse_args()
    
    model, images = prepare_model(args.model, args.dataset)
    tiles = read(args.shapes, args.shapes_groups)
    outname = args.output
    if not (outname.endswith(".png") or outname.endswith(".jpg")):
        outname += ".png"
    outputpath = "../results/photomosaic/" 
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    canvas = paint(tiles, model, images, canvas_size = (224, 224, 3), path = outputpath + outname)
