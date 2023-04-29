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
from torch.optim.lr_scheduler import StepLR
import os
import pickle

# Load the best parameters
if os.path.exists("clip_best_params.pkl"):
    print("Loading best parameters...")
    best_params = pickle.load(open("clip_best_params.pkl", "rb"))
    print("Best parameters: ")
    print(best_params)

    delta_lr = best_params["delta_lr"]
    angle_lr = best_params["angle_lr"]
    tranlation_lr = best_params["tranlation_lr"]
    color_lr = best_params["color_lr"]

    neg_clip_coe = best_params["neg_clip_coe"]

    coe_delta = torch.tensor(
        [best_params["reg_delta_coe_x"], best_params["reg_delta_coe_y"]],
        dtype=torch.float32,
    )
    coe_displacement = torch.tensor(
        [best_params["reg_displacement_coe_x"], best_params["reg_displacement_coe_y"]],
        dtype=torch.float32,
    )
    coe_angle = torch.tensor(best_params["angle_coe"], dtype=torch.float32)

    coe_image = torch.tensor(best_params["image_coe"], dtype=torch.float32)

    coe_distance = torch.tensor(best_params["distance_coe"], dtype=torch.float32)
else:
    print("No best parameters found, using default parameters...")
    delta_lr = 0.01
    angle_lr = 0.01
    tranlation_lr = 0.01
    color_lr = 0.01

    neg_clip_coe = 0.3

    coe_delta = torch.tensor([1e-4, 1e-4], dtype=torch.float32)
    coe_displacement = torch.tensor([0.0, 0.0], dtype=torch.float32)
    coe_angle = torch.tensor(0.0, dtype=torch.float32)

    coe_image = torch.tensor(0.0, dtype=torch.float32)

    coe_distance = torch.tensor(1e-4, dtype=torch.float32)

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

scene_args = pydiffvg.RenderFunction.serialize_scene(
    canvas_width, canvas_height, shapes, shape_groups
)
img = render(
    canvas_width,  # width
    canvas_height,  # height
    2,  # num_samples_x
    2,  # num_samples_y
    1,  # seed
    None,  # background_image
    *scene_args
)
pydiffvg.imwrite(img.cpu(), "results/clip/init.png", gamma=gamma)

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
num_interations = 50
scheduler_delta = StepLR(optimizer_delta, step_size=num_interations // 3, gamma=0.5)
scheduler_angle = StepLR(optimizer_angle, step_size=num_interations // 3, gamma=0.5)
scheduler_translation = StepLR(
    optimizer_translation, step_size=num_interations // 3, gamma=0.5
)
scheduler_color = StepLR(optimizer_color, step_size=num_interations // 3, gamma=0.5)

def calc_loss(shapes, shape_groups, with_reg=False):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        num_interations , # seed
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
    
    if with_reg:
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
    else:
        # without regularization
        loss = (
            pos_clip_loss
            + neg_clip_loss
        )
        
    return loss

def render_scene(name, shapes, shape_groups):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        num_interations ,  # seed
        None,  # background_image
        *scene_args
    )
    pydiffvg.imwrite(
        img.cpu(), "results/postprocess/output-{}.png".format(name), gamma=gamma
    )

def rec2x(ori_shapes, ori_shape_groups):
    render_scene("before", ori_shapes, ori_shape_groups)
    origin_loss = calc_loss(ori_shapes, ori_shape_groups)
    print("Origin loss = ", origin_loss)

    # render_scene("after", ori_shapes[:-1], ori_shape_groups[:-1])
    # render_scene("after2", ori_shapes, ori_shape_groups)
    # return

    for (id, (rect, rect_group)) in enumerate(zip(ori_shapes, ori_shape_groups)):

        with torch.no_grad():
            ori_shapes[id].delta += torch.tensor([5, 5])
            ori_shapes[id].update()
        cur_loss = calc_loss(ori_shapes, ori_shape_groups)
        print("cur loss = ", cur_loss)
        if (cur_loss < origin_loss):
            print("update")
            # origin_loss = cur_loss
        else:
            with torch.no_grad():
                ori_shapes[id].delta += torch.tensor([-5, -5])
                ori_shapes[id].update()
    render_scene("after", ori_shapes, ori_shape_groups)

def recdel(ori_shapes, ori_shape_groups):
    origin_loss = calc_loss(ori_shapes, ori_shape_groups)
    print("Origin loss = ", origin_loss)

    loss_contrib = []

    for (id, (rect, rect_group)) in enumerate(zip(ori_shapes, ori_shape_groups)):
        shapes = ori_shapes.copy()
        shapes.remove(rect)
        # print(shapes)
        # print(type(shapes))
        shape_groups = ori_shape_groups.copy()
        shape_groups.remove(rect_group)
        
        for id_ in range(id, len(shapes)):
            # print(shapes[id_].id)
            # print(shape_groups[id_].id)
            # print(type(shapes[id_].id))
            # shapes[id_].id = str(int(shapes[id_].id)-1)
            # print("shapes_groupsID: ", shape_groups[id_].shape_ids)
            # print(torch.tensor([int(shape_groups[id_].shape_ids[0]) - 1]))
            shape_groups[id_].shape_ids = torch.tensor([int(shape_groups[id_].shape_ids[0])-1])

        print("calculating loss...")
        cur_loss = calc_loss(shapes, shape_groups)
        print("cur loss = ", cur_loss)
        loss_contrib.append((id, - cur_loss + origin_loss))

    print(loss_contrib)



# rec2x(shapes, shape_groups)
recdel(shapes, shape_groups)
quit()

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

    # Save the intermediate render.
    if t % 5 == 0:
        pydiffvg.imwrite(
            img.cpu(), "results/clip/iter_{}.png".format(t // 5), gamma=gamma
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
    optimizer_color.step()
    optimizer_delta.step()
    optimizer_angle.step()
    optimizer_translation.step()

    # Take a scheduler step in the learning rate.
    scheduler_color.step()
    scheduler_delta.step()
    scheduler_angle.step()
    scheduler_translation.step()

rec2x(shapes, shape_groups)

# Render the final result.
scene_args = pydiffvg.RenderFunction.serialize_scene(
    canvas_width, canvas_height, shapes, shape_groups
)
img = render(
    canvas_width,  # width
    canvas_height,  # height
    2,  # num_samples_x
    2,  # num_samples_y
    102,  # seed
    None,  # background_image
    *scene_args
)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), "results/clip/final.png", gamma=gamma)

# Convert the intermediate renderings to a video.
call(
    [
        "ffmpeg",
        "-framerate",
        "24",
        "-i",
        "results/clip/iter_%d.png",
        "-vb",
        "20M",
        "results/clip/out.mp4",
    ]
)

# TODO: Fetch top k rectangles which contributes the most to the loss for image replacement
