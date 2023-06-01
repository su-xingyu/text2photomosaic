## Mosiac Generation

This folder contains all scripts for mosaic generation.

### Text-to-mosiac Generation

#### clip_find_best_params.py

`clip_find_best_params.py` is the CLIP optimization loop wrapped with **Optuna**, provided with an input prompt, it will generate a `clip_best_params.pkl` under `../results/clip/pkls` indicating the best hyper-parameter settings for this input.

Example usage:

```
python clip_find_best_params.py \
    --prompt "a red heart" \
    --num_interations 1000
```

#### clip_best_params.py

`clip_best_params.py` is the original file for text-to-mosaic generation. It consists of 2 stages

1. The optimization loop whose itermediate results will be written to `../results/clip`, and the scene description file will be written to `../results/clip/pkls`.
2. 2 post-processing stage whose results will be written to `../results/clip`, and the scene description file will be written to `../results/clip/pkls`.

When a `clip_best_params.pkl` is detected under `../results/clip/pkls`, this script will use the best configurations. Otherwise it will use the hyper-paramters specified in command line.

Example usage:

```
python clip_best_params.py \
    --prompt "a red heart" \
    --neg_clip_coe 0.3 \
    --delta_coe_x 1e-4 \
    --delta_coe_y 1e-4 \
    --displacement_coe_x 0.0 \
    --displacement_coe_y 0.0 \
    --angle_coe 0.0 \
    --image_coe 0.0 \
    --overlap_coe 0.0 \
    --neighbor_num 0 \
    --neighbor_coe 0.0 \
    --joint_coe 1e-4
```

You can refer to `input/experiments` for more examples.

### Target-to-mosiac Generation

#### target_find_best_params.py

`target_find_best_params.py` is the optimization loop wrapped with **Optuna**, provided with an input target image, it will generate a `target_best_params.pkl` under `../results/target/pkls` indicating the best hyper-parameter settings for this input.

Example usage:

```
python target_find_best_params.py \
    --target_image "inputs/target_exp1.png" \
    --num_interations 1000
```

#### target_best_params.py

`target_best_params.py` is the original file for Diffvg-based target-to-mosaic generation. It contains the optimization loop whose itermediate results will be written to `../results/target`, and the scene description file will be written to `../results/target/pkls`.

When a `target_best_params.pkl` is detected under `../results/target/pkls`, this script will use the best configurations. Otherwise it will use the hyper-paramters specified in command line.

Example usage:

```
python target_best_params.py \
    --target_image "inputs/target_exp1.png" \
    --delta_coe_x 1e-4 \
    --delta_coe_y 1e-4 \
    --displacement_coe_x 1e-3 \
    --displacement_coe_y 1e-3 \
    --angle_coe 0.0 \
    --overlap_coe 0.0 \
    --neighbor_num 0 \
    --neighbor_coe 0.0 \
    --joint_coe 0.0
```

You can refer to `input/experiments` for more examples.

#### target_naive.py

`target_naive.py` is the naive implementation of target-to-mosaic generation, whose result will be written to `../results/target`.

Example usage:

```
python target_naive.py \
    --target_image "inputs/target_exp1.png"
```