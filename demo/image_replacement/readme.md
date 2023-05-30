`retrieve/` folder contains the moduls for image retrieving
`model.pkl` is the photo retrieving model, generated according to the given dataset

To run demo of converting mosaic image to photomosaic image, simply use `python demo_replace.py`
The parameters could be explained by appending `--help`

`--model` path to image retrieving model, the default model is provided
`--dataset` path to dataset, default dataset is in `retrieve/dataset_demo`
`--output` name of generated photomosaic image, saved under the path `../results/photomosaic/`

The following two parameter are used in pair, contains the data of one mosaic image
`--shapes` 
`--shapes_groups`
