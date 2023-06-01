# Text2Photomosaic

This is the course project for CS-413 Computational Photography @ EPFL.

## Prepare the Environment

You will need to setup environment for both [Diffvg](https://github.com/BachiLi/diffvg) and [CLIP](https://github.com/openai/CLIP). Go to each project's repository for detailed instructions of environment setup.

Or you can refer to the **Pre Installation** part of [CLIPDraw](https://colab.research.google.com/github/kvfrans/clipdraw/blob/main/clipdraw.ipynb), which should also work.

## Quick Start

**Note:** You can refer to each subfolder for more detailed instructions.

### Text-to-photomosaic generation

1. Run the following command

```
python demo/clip_best_params.py
```

This will generate mosaic image for a default prompt "a red heart". Result images will be stored at `demo/results/clip`

Once the job above finishes, you will obtain `demo/results/clip/pkls/clip_shapes.pkl` and `demo/results/clip/pkls/clip_shape_groups.pkl`. Then run

```
python demo/image_replacement/demo_replace.py \
    --shapes "demo/results/clip/pkls/clip_shapes.pkl" \
    --shape_groups `demo/results/clip/pkls/clip_shape_groups.pkl`
```

This will generate the corresponding photomosaic image, which will be stored at `demo/results/photomosaic`.

### Target-to-photomosaic generation

The workflow is similar to `text-to-photomosaic generation`. But you should run the following commands sequentially

```
python demo/target_best_params.py

python demo/image_replacement/demo_replace.py \
    --shapes "demo/results/target/pkls/target_shapes.pkl" \
    --shape_groups `demo/results/target/pkls/target_shape_groups.pkl`
```

## Results

### Text-to-photomosaic generation

**Prompt:** "a red heart"

**Left:** mosiac image, **Right:** photomosaic image

<p float="left">
  <img src="/demo/results/previous_results/clip/exp1/after_delete.png" width="224" />
  <img src="/demo/results/previous_results/clip/exp1/photomosaic.png" width="224" />
</p>

**Other text-to-mosaic results**

**Left:** a green tree in the desert, **Right:** a flower on a rock

<p float="left">
  <img src="/demo/results/previous_results/clip/exp2/after_optimization.png" width="224" />
  <img src="/demo/results/previous_results/clip/exp3/after_optimization.png" width="224" />
</p>

### Target-to-photomosaic generation

**Left:** target image, **Right:** mosiac image

<p float="left">
  <img src="/demo/mosaic_generation/inputs/target_exp1.png" width="224" />
  <img src="/demo/results/previous_results/target/exp1/diffvg_0.07319.png" width="224" /> 
</p>