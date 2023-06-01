from subprocess import call
import pydiffvg
import torch
from my_shape import PolygonRect, RotationalShapeGroup
from utils import (
    cal_loss,
    postprocess_delete_rect,
    postprocess_scale_rect,
    render_image,
)
import torchvision.transforms as transforms
import clip
from torch.optim.lr_scheduler import StepLR
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt", help="prompt for mosaic generation", default="a red heart"
)
parser.add_argument(
    "--num_interations",
    help="number of optimization iterations",
    default=1000,
    type=int,
)
parser.add_argument("--neg_clip_coe", help="neg_clip_coe", default=0.3, type=float)
parser.add_argument("--delta_coe_x", help="delta_coe_x", default=1e-4, type=float)
parser.add_argument("--delta_coe_y", help="delta_coe_y", default=1e-4, type=float)
parser.add_argument(
    "--displacement_coe_x", help="displacement_coe_x", default=0.0, type=float
)
parser.add_argument(
    "--displacement_coe_y", help="displacement_coe_y", default=0.0, type=float
)
parser.add_argument("--angle_coe", help="angle_coe", default=0.0, type=float)
parser.add_argument("--image_coe", help="image_coe", default=0.0, type=float)
parser.add_argument("--overlap_coe", help="overlap_coe", default=1e-4, type=float)
parser.add_argument("--neighbor_num", help="neighbor_num", default=0, type=int)
parser.add_argument("--neighbor_coe", help="neighbor_coe", default=0.0, type=float)
parser.add_argument("--joint_coe", help="joint_coe", default=1e-4, type=float)
args = parser.parse_args()

RESULTS_PATH = "../results/clip/"
PKLS_PATH = os.path.join(RESULTS_PATH, "pkls")

# Create folder for saving results
if not os.path.exists(PKLS_PATH):
    print("Creating folder for saving results...")
    os.makedirs(PKLS_PATH)

# Load the best parameters
if os.path.exists(os.path.join(PKLS_PATH, "clip_best_params.pkl")):
    print("Loading best parameters...")
    best_params = pickle.load(
        open(os.path.join(PKLS_PATH, "clip_best_params.pkl"), "rb")
    )
    print("Best parameters: ")
    print(best_params)

    delta_lr = best_params["delta_lr"]
    angle_lr = best_params["angle_lr"]
    tranlation_lr = best_params["tranlation_lr"]
    color_lr = best_params["color_lr"]

    neg_clip_coe = best_params["neg_clip_coe"]

    delta_coe = torch.tensor(
        [best_params["reg_delta_coe_x"], best_params["reg_delta_coe_y"]],
        dtype=torch.float32,
    )
    displacement_coe = torch.tensor(
        [best_params["reg_displacement_coe_x"], best_params["reg_displacement_coe_y"]],
        dtype=torch.float32,
    )
    angle_coe = torch.tensor(best_params["angle_coe"], dtype=torch.float32)

    image_coe = torch.tensor(best_params["image_coe"], dtype=torch.float32)

    overlap_coe = torch.tensor(best_params["overlap_coe"], dtype=torch.float32)

    neighbor_num = best_params["neighbor_num"]
    neighbor_coe = torch.tensor(best_params["neighbor_coe"], dtype=torch.float32)

    joint_coe = torch.tensor(best_params["joint_coe"], dtype=torch.float32)
else:
    print("No best parameters found, using default parameters...")
    delta_lr = 0.01
    angle_lr = 0.01
    tranlation_lr = 0.01
    color_lr = 0.01

    neg_clip_coe = args.neg_clip_coe

    delta_coe = torch.tensor([args.delta_coe_x, args.delta_coe_y], dtype=torch.float32)
    displacement_coe = torch.tensor(
        [args.displacement_coe_x, args.displacement_coe_y], dtype=torch.float32
    )
    angle_coe = torch.tensor(args.angle_coe, dtype=torch.float32)

    image_coe = torch.tensor(args.image_coe, dtype=torch.float32)

    overlap_coe = torch.tensor(args.overlap_coe, dtype=torch.float32)

    neighbor_num = args.neighbor_num
    neighbor_coe = torch.tensor(args.neighbor_coe, dtype=torch.float32)

    joint_coe = torch.tensor(args.joint_coe, dtype=torch.float32)

coe_dict = {
    "neg_clip_coe": neg_clip_coe,
    "delta_coe": delta_coe,
    "displacement_coe": displacement_coe,
    "angle_coe": angle_coe,
    "image_coe": image_coe,
    "overlap_coe": overlap_coe,
    "neighbor_num": neighbor_num,
    "neighbor_coe": neighbor_coe,
    "joint_coe": joint_coe,
    "threshold": "mean",
}

# Initialize CLIP text input
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device, jit=False)

prompt = args.prompt
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
            coe_trans=torch.tensor([canvas_width, canvas_height], dtype=torch.float32),
        )
        shape_groups.append(rect_group)

for rect in shapes:
    rect.update()
for rect_group in shape_groups:
    rect_group.update()

img = render_image(canvas_width, canvas_height, shapes, shape_groups, render, seed=1)
pydiffvg.imwrite(img.cpu(), os.path.join(RESULTS_PATH, "init.png"), gamma=gamma)

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

# Run Adam iterations.
num_interations = args.num_interations
scheduler_delta = StepLR(optimizer_delta, step_size=num_interations // 3, gamma=0.5)
scheduler_angle = StepLR(optimizer_angle, step_size=num_interations // 3, gamma=0.5)
scheduler_translation = StepLR(
    optimizer_translation, step_size=num_interations // 3, gamma=0.5
)
scheduler_color = StepLR(optimizer_color, step_size=num_interations // 3, gamma=0.5)

for t in range(num_interations):
    print("Optimization iteration:", t)

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

    # Save the intermediate render.
    if t % 5 == 0:
        pydiffvg.imwrite(
            img.cpu(),
            os.path.join(RESULTS_PATH, "iter_{}.png".format(t // 5)),
            gamma=gamma,
        )

    loss, _ = cal_loss(
        img,
        shapes,
        shape_groups,
        model,
        text_features,
        coe_dict,
        use_aug=True,
        augment_trans=augment_trans,
        use_neg=True,
        text_features_neg=text_features_neg,
        verbose=True,
    )

    # Backpropagate the gradients.
    loss.backward(retain_graph=True)

    # Take a gradient descent step.
    optimizer_color.step()
    optimizer_delta.step()
    optimizer_angle.step()
    optimizer_translation.step()

    # Take a scheduler step in the learning rate.
    scheduler_color.step()
    scheduler_delta.step()
    scheduler_angle.step()
    scheduler_translation.step()

img = render_image(canvas_width, canvas_height, shapes, shape_groups, render, seed=102)
pydiffvg.imwrite(
    img.cpu(), os.path.join(RESULTS_PATH, "after_optimization.png"), gamma=gamma
)

pickle.dump(shapes, open(os.path.join(PKLS_PATH, "clip_shapes_no_pp.pkl"), "wb"))
pickle.dump(
    shape_groups, open(os.path.join(PKLS_PATH, "clip_shape_groups_no_pp.pkl"), "wb")
)

# We care only about pos_clip_loss when doing post-processing
postprocess_delete_rect(
    canvas_width,
    canvas_height,
    render,
    shapes,
    shape_groups,
    model,
    text_features,
    verbose=True,
)

img = render_image(canvas_width, canvas_height, shapes, shape_groups, render, seed=102)
pydiffvg.imwrite(img.cpu(), os.path.join(RESULTS_PATH, "after_delete.png"), gamma=gamma)

postprocess_scale_rect(
    canvas_width,
    canvas_height,
    render,
    shapes,
    shape_groups,
    model,
    text_features,
    scale=1.2,
    max_iter=100,
    verbose=True,
)

img = render_image(canvas_width, canvas_height, shapes, shape_groups, render, seed=102)
pydiffvg.imwrite(img.cpu(), os.path.join(RESULTS_PATH, "after_scale.png"), gamma=gamma)


# Render the final result.
img = render_image(canvas_width, canvas_height, shapes, shape_groups, render, seed=102)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), os.path.join(RESULTS_PATH, "final.png"), gamma=gamma)

pickle.dump(shapes, open(os.path.join(PKLS_PATH, "clip_shapes.pkl"), "wb"))
pickle.dump(shape_groups, open(os.path.join(PKLS_PATH, "clip_shape_groups.pkl"), "wb"))

# Convert the intermediate renderings to a video.
call(
    [
        "ffmpeg",
        "-framerate",
        "24",
        "-i",
        os.path.join(RESULTS_PATH, "iter_%d.png"),
        "-vb",
        "20M",
        os.path.join(RESULTS_PATH, "out.mp4"),
    ]
)
