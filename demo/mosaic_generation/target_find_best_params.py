import pydiffvg
import torch
from my_shape import PolygonRect, RotationalShapeGroup
from utils import (
    diffvg_regularization_term,
    pairwise_diffvg_regularization_term,
    joint_regularization_term,
    render_image,
)
import optuna
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target_image", help="path to target image", default="inputs/target_exp1.png"
)
args = parser.parse_args()

RESULTS_PATH = "../results/target/"
PKLS_PATH = os.path.join(RESULTS_PATH, "pkls")

# Create folder for saving results
if not os.path.exists(PKLS_PATH):
    print("Creating folder for saving results...")
    os.makedirs(PKLS_PATH)

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

gamma = 2.2
render = pydiffvg.RenderFunction.apply

# Load target image
target = Image.open(args.target_image)
target = (torch.from_numpy(np.array(target)).float() / 255.0) ** gamma
target = target[:, :, 3:4] * target[:, :, :3] + torch.ones(
    target.shape[0], target.shape[1], 3, device=pydiffvg.get_device()
) * (1 - target[:, :, 3:4])
target = target[:, :, :3]
canvas_width, canvas_height = target.shape[1], target.shape[0]

# Optuna trail


def objective(trial):
    delta_lr = trial.suggest_float("delta_lr", 1e-4, 1e-1, log=True)
    angle_lr = trial.suggest_float("angle_lr", 1e-4, 1e-1, log=True)
    tranlation_lr = trial.suggest_float("tranlation_lr", 1e-4, 1e-1, log=True)
    color_lr = trial.suggest_float("color_lr", 1e-4, 1e-1, log=True)

    reg_delta_coe_x = trial.suggest_float("reg_delta_coe_x", 1e-6, 1.0, log=True)
    reg_delta_coe_y = trial.suggest_float("reg_delta_coe_y", 1e-6, 1.0, log=True)
    delta_coe = torch.tensor([reg_delta_coe_x, reg_delta_coe_y], dtype=torch.float32)

    reg_displacement_coe_x = trial.suggest_float(
        "reg_displacement_coe_x", 1e-6, 1.0, log=True
    )
    reg_displacement_coe_y = trial.suggest_float(
        "reg_displacement_coe_y", 1e-6, 1.0, log=True
    )
    displacement_coe = torch.tensor(
        [reg_displacement_coe_x, reg_displacement_coe_y], dtype=torch.float32
    )

    angle_coe = torch.tensor(trial.suggest_float("angle_coe", 1e-6, 1.0, log=True))

    overlap_coe = trial.suggest_float("overlap_coe", 1e-6, 1.0, log=True)

    neighbor_num = trial.suggest_int("neighbor_num", 1, 8)
    neighbor_coe = trial.suggest_float("neighbor_coe", 1e-6, 1.0, log=True)

    joint_coe = torch.tensor(trial.suggest_float("joint_coe", 1e-6, 1.0, log=True))

    # Initializations
    shapes = []
    shape_groups = []
    for x in range(0, canvas_width, canvas_width // 10):
        for y in range(0, canvas_height, canvas_height // 10):
            rect = PolygonRect(
                upper_left=torch.tensor([x, y]),
                width=canvas_width // 10 + 0.0,
                height=canvas_height // 10 + 0.0,
            )
            shapes.append(rect)
            rect_group = RotationalShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=torch.cat([torch.rand(3), torch.tensor([1.0])]),
                transparent=False,
                coe_ang=torch.tensor(1.0),
                coe_trans=torch.tensor(
                    [canvas_width, canvas_height], dtype=torch.float32
                ),
            )
            shape_groups.append(rect_group)

    optimizer_delta = torch.optim.Adam([rect.delta for rect in shapes], lr=delta_lr)
    optimizer_angle = torch.optim.Adam(
        [rect_group.angle for rect_group in shape_groups], lr=angle_lr
    )
    optimizer_translation = torch.optim.Adam(
        [rect_group.translation for rect_group in shape_groups], lr=tranlation_lr
    )
    optimizer_color = torch.optim.Adam(
        [rect_group.color for rect_group in shape_groups], lr=color_lr
    )

    num_interations = 1000
    scheduler_delta = StepLR(optimizer_delta, step_size=num_interations // 3, gamma=0.5)
    scheduler_angle = StepLR(optimizer_angle, step_size=num_interations // 3, gamma=0.5)
    scheduler_translation = StepLR(
        optimizer_translation, step_size=num_interations // 3, gamma=0.5
    )
    scheduler_color = StepLR(optimizer_color, step_size=num_interations // 3, gamma=0.5)

    # Run optimization iterations.
    for t in range(num_interations):
        print("iteration:", t)

        optimizer_delta.zero_grad()
        optimizer_angle.zero_grad()
        optimizer_translation.zero_grad()
        optimizer_color.zero_grad()

        for rect in shapes:
            rect.update()
        for rect_group in shape_groups:
            rect_group.update()

        img = render_image(
            canvas_width, canvas_height, shapes, shape_groups, render, seed=t + 1
        )

        # Pixel-wise loss.
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        pixel_loss = torch.sum((img - target) ** 2) / (canvas_width * canvas_height)

        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Regularization term
        diffvg_regularization_loss = torch.zeros(1, device=pydiffvg.get_device())
        if (
            torch.norm(delta_coe) > 0
            or torch.norm(displacement_coe) > 0
            or torch.norm(angle_coe) > 0
        ):
            diffvg_regularization_loss = diffvg_regularization_term(
                shapes,
                shape_groups,
                coe_delta=delta_coe,
                coe_displacement=displacement_coe,
                coe_angle=angle_coe,
            )
        pairwise_diffvg_regularization_loss = torch.zeros(
            1, device=pydiffvg.get_device()
        )
        if torch.norm(overlap_coe) > 0 or torch.norm(neighbor_coe) > 0:
            pairwise_diffvg_regularization_loss = pairwise_diffvg_regularization_term(
                shapes,
                shape_groups,
                coe_overlap=overlap_coe,
                num_neighbor=neighbor_num,
                coe_neighbor=neighbor_coe,
            )
        joint_regularization_loss = torch.zeros(1, device=pydiffvg.get_device())
        if torch.norm(joint_coe) > 0:
            joint_regularization_loss = joint_regularization_term(
                shapes,
                shape_groups,
                img,
                num_neighbor=1,
                coe_joint=joint_coe,
                threshold="max",
            )
        loss = (
            pixel_loss
            + diffvg_regularization_loss
            + pairwise_diffvg_regularization_loss
            + joint_regularization_loss
        )

        print("pixel_loss:", pixel_loss.item())
        print("diffvg_regularization_loss:", diffvg_regularization_loss.item())
        print(
            "pairwise_diffvg_regularization_loss:",
            pairwise_diffvg_regularization_loss.item(),
        )
        print("loss:", loss.item())

        # Backpropagate the gradients.
        loss.backward(retain_graph=True)

        # Take a gradient descent step.
        optimizer_delta.step()
        optimizer_angle.step()
        optimizer_translation.step()
        optimizer_color.step()

        # Take a scheduler step in the learning rate.
        scheduler_delta.step()
        scheduler_angle.step()
        scheduler_translation.step()
        scheduler_color.step()

    return pixel_loss.item()


study = optuna.create_study()
study.optimize(objective, n_trials=2)
with open(os.path.join(PKLS_PATH, "target_best_params.pkl"), "wb") as f:
    pickle.dump(study.best_params, f)
