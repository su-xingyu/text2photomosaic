from subprocess import call
import pydiffvg
import torch
from my_shape import PolygonRect, RotationalShapeGroup
from utils import (
    diffvg_regularization_term,
    pairwise_diffvg_regularization_term,
    image_regularization_term,
)
import torchvision.transforms as transforms
import clip
import optuna
from torch.optim.lr_scheduler import StepLR
import pickle

# Initialize CLIP text input
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device, jit=False)

prompt = "a red heart"
neg_prompt = "an ugly, messy picture."
text_input = clip.tokenize(prompt).to(device)
text_input_neg = clip.tokenize(neg_prompt).to(device)
use_neg = True

with torch.no_grad():
    text_features = model.encode_text(text_input)
    text_features_neg = model.encode_text(text_input_neg)

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

gamma = 1.0
render = pydiffvg.RenderFunction.apply

canvas_width, canvas_height = 224, 224

# Image Augmentation Transformation
augment_trans = transforms.Compose(
    [
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

# Optuna trail


def objective(trial):
    delta_lr = trial.suggest_float("delta_lr", 1e-4, 1e-1, log=True)
    angle_lr = trial.suggest_float("angle_lr", 1e-4, 1e-1, log=True)
    tranlation_lr = trial.suggest_float("tranlation_lr", 1e-4, 1e-1, log=True)
    color_lr = trial.suggest_float("color_lr", 1e-4, 1e-1, log=True)

    neg_clip_coe = trial.suggest_float("neg_clip_coe", 0.0, 0.5)

    reg_delta_coe_x = trial.suggest_float("reg_delta_coe_x", 1e-6, 1.0, log=True)
    reg_delta_coe_y = trial.suggest_float("reg_delta_coe_y", 1e-6, 1.0, log=True)
    coe_delta = torch.tensor([reg_delta_coe_x, reg_delta_coe_y], dtype=torch.float32)

    reg_displacement_coe_x = trial.suggest_float(
        "reg_displacement_coe_x", 1e-6, 1.0, log=True
    )
    reg_displacement_coe_y = trial.suggest_float(
        "reg_displacement_coe_y", 1e-6, 1.0, log=True
    )
    coe_displacement = torch.tensor(
        [reg_displacement_coe_x, reg_displacement_coe_y], dtype=torch.float32
    )

    angle_coe = trial.suggest_float("angle_coe", 1e-6, 1.0, log=True)
    coe_angle = torch.tensor(angle_coe, dtype=torch.float32)

    coe_image = trial.suggest_float("image_coe", 1e-6, 1.0, log=True)

    coe_distance = trial.suggest_float("distance_coe", 1e-6, 1.0, log=True)

    # Initializations
    shapes = []
    shape_groups = []
    for x in range(0, 224, 16):
        for y in range(0, 224, 16):
            rect = PolygonRect(upper_left=torch.tensor([x, y]), width=14.0, height=14.0)
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

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = render(
            canvas_width,  # width
            canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            t + 1,  # seed
            None,  # background_image
            *scene_args
        )

        # Transform image for CLIP input
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Compute the loss
        pos_clip_loss = 0
        neg_clip_loss = 0
        NUM_AUGS = 4
        img_augs = []
        for n in range(NUM_AUGS):
            img_augs.append(augment_trans(img))
        img_batch = torch.cat(img_augs)
        image_features = model.encode_image(img_batch)
        for n in range(NUM_AUGS):
            pos_clip_loss -= torch.cosine_similarity(
                text_features, image_features[n : n + 1], dim=1
            )
            if use_neg:
                neg_clip_loss += (
                    torch.cosine_similarity(
                        text_features_neg, image_features[n : n + 1], dim=1
                    )
                    * neg_clip_coe
                )

        # Regularization term
        diffvg_regularization_loss = diffvg_regularization_term(
            shapes,
            shape_groups,
            coe_delta=coe_delta,
            coe_displacement=coe_displacement,
            coe_angle=coe_angle,
        )
        pairwise_diffvg_regularization_loss = pairwise_diffvg_regularization_term(
            shapes, shape_groups, coe_distance=coe_distance
        )
        image_regularization_loss = image_regularization_term(img, coe_image=coe_image)
        loss = (
            pos_clip_loss
            + neg_clip_loss
            + diffvg_regularization_loss
            + pairwise_diffvg_regularization_loss
            + image_regularization_loss
        )

        print("pos_clip_loss:", pos_clip_loss.item())
        print("neg_clip_loss:", neg_clip_loss.item())
        print("diffvg_regularization_loss:", diffvg_regularization_loss.item())
        print(
            "pairwise_diffvg_regularization_loss:",
            pairwise_diffvg_regularization_loss.item(),
        )
        print("image_regularization_loss:", image_regularization_loss.item())
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

    return pos_clip_loss.item()


study = optuna.create_study()
study.optimize(objective, n_trials=2)
with open("clip_best_params.pkl", "wb") as f:
    pickle.dump(study.best_params, f)
