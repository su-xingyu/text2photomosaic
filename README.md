# Text2Photomosaic

## Prepare the Environment
Refer to the **Pre Installation** part of https://colab.research.google.com/github/kvfrans/clipdraw/blob/main/clipdraw.ipynb

## Run Demo
To generate the mosaic image, run following code
```
python demo/clip_best_params.py   # defaultly run 500 iterations
python demo/clip_best_params.py --num_iter=NUM_ITER
```
To replace mosaic tiles with images
```
python demo/demo_replace.py  # use --help to learn more about the arguments
```