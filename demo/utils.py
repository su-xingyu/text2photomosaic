import torch
import torchvision.transforms as transforms
import pydiffvg

TWO_PI = 2 * torch.pi


# ----------------------- Loss calculation -----------------------


def diffvg_regularization_term(
    shapes,
    shape_groups,
    coe_delta=torch.tensor([1.0, 1.0]),
    coe_displacement=torch.tensor([1.0, 1.0]),
    coe_angle=torch.tensor(1.0),
):
    regularization_term = 0
    for shape, shape_group in zip(shapes, shape_groups):
        # Delta regularization term
        regularization_term += torch.sum(coe_delta * ((shape.delta / shape.size) ** 2))

        # Displacement regularization term
        center = shape.upper_left + shape.size / 2
        center_transformed = center + shape.delta / 2
        center_transformed = torch.matmul(
            shape_group.shape_to_canvas,
            torch.cat((center_transformed, torch.tensor([1.0]))),
        )[:2]
        displacement = center_transformed - center
        regularization_term += torch.sum(
            coe_displacement * ((displacement / shape.size) ** 2)
        )

        # Angle regularization term
        regularization_term += coe_angle * ((shape_group.angle / TWO_PI) ** 2)

    return regularization_term


def pairwise_diffvg_regularization_term(
    shapes,
    shape_groups,
    coe_overlap=torch.tensor(1.0),
    num_neighbor=1,
    coe_neighbor=torch.tensor(1.0),
    threshold="mean",
):
    centers_transformed = torch.stack(
        [
            torch.matmul(
                shape_group.shape_to_canvas,
                torch.cat(
                    (
                        (shape.upper_left + shape.size / 2 + shape.delta / 2),
                        torch.tensor([1.0]),
                    )
                ),
            )[:2]
            for shape, shape_group in zip(shapes, shape_groups)
        ],
        dim=-1,
    )
    # Half of the sides
    if threshold == "mean":
        sides_transformed = torch.stack(
            [torch.mean(shape.size / 2 + shape.delta / 2) for shape in shapes], dim=-1
        )
    elif threshold == "max":
        sides_transformed = torch.stack(
            [torch.max(shape.size / 2 + shape.delta / 2) for shape in shapes], dim=-1
        )
    elif threshold == "diagonal":
        sides_transformed = torch.stack(
            [torch.norm(shape.size / 2 + shape.delta / 2) for shape in shapes], dim=-1
        )
    rolling_centers_transformed = torch.stack(
        [
            torch.roll(centers_transformed, i, dims=-1)
            for i in range(1, centers_transformed.shape[-1])
        ],
        dim=-1,
    )
    rolling_sides_transformed = torch.stack(
        [
            torch.roll(sides_transformed, i, dims=-1)
            for i in range(1, sides_transformed.shape[-1])
        ],
        dim=-1,
    )
    pairwise_distances = torch.norm(
        centers_transformed.unsqueeze(-1) - rolling_centers_transformed, dim=0
    )
    # Sum up half of the sides of the two rectangles
    pairwise_sum_sides = sides_transformed.unsqueeze(-1) + rolling_sides_transformed
    normalization_term = torch.tensor(
        [torch.mean(shape.size) for shape in shapes]
    ).unsqueeze(-1)

    regularization_term = 0

    # Overlap regularization term
    regularization_term += coe_overlap * torch.sum(
        (
            torch.nn.functional.relu(pairwise_sum_sides - pairwise_distances)
            / normalization_term
        )
        ** 2
    )

    # Neighbor regularization term (neighbors not too far apart)
    neighbor_distances, _ = torch.topk(
        torch.nn.functional.relu(2 * pairwise_distances - pairwise_sum_sides)
        / normalization_term,
        k=num_neighbor,
        dim=-1,
        largest=False,
    )
    regularization_term += coe_neighbor * torch.sum(neighbor_distances**2)

    return regularization_term


def image_regularization_term(image, coe_image=torch.tensor(1.0)):
    gray_image = transforms.functional.rgb_to_grayscale(image)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    grad_x = (
        torch.nn.functional.conv2d(gray_image, sobel_x.unsqueeze(0).unsqueeze(0)) / 4
    )
    grad_y = (
        torch.nn.functional.conv2d(gray_image, sobel_y.unsqueeze(0).unsqueeze(0)) / 4
    )
    regularization_term = coe_image * torch.sqrt(torch.mean(grad_x**2 + grad_y**2))

    return regularization_term


def joint_regularization_term(
    shapes,
    shape_groups,
    image,
    num_neighbor=1,
    coe_joint=torch.tensor(1.0),
    threshold="mean",
):
    # For each pixel, check whether it is cover by the closest rectangles
    width, height = image.shape[-2:]
    x_coords = torch.linspace(0, width - 1, width).repeat(height, 1)
    y_coords = x_coords.t()
    coords = torch.stack((x_coords, y_coords), dim=0)

    centers_transformed = torch.stack(
        [
            torch.matmul(
                shape_group.shape_to_canvas,
                torch.cat(
                    (
                        (shape.upper_left + shape.size / 2 + shape.delta / 2),
                        torch.tensor([1.0]),
                    )
                ),
            )[:2]
            for shape, shape_group in zip(shapes, shape_groups)
        ],
        dim=-1,
    )

    # Half of the sides
    if threshold == "mean":
        sides_transformed = torch.stack(
            [torch.mean(shape.size / 2 + shape.delta / 2) for shape in shapes], dim=-1
        )[:, None, None]
    elif threshold == "max":
        sides_transformed = torch.stack(
            [torch.max(shape.size / 2 + shape.delta / 2) for shape in shapes], dim=-1
        )[:, None, None]
    elif threshold == "diagonal":
        sides_transformed = torch.stack(
            [torch.norm(shape.size / 2 + shape.delta / 2) for shape in shapes], dim=-1
        )[:, None, None]
    normalization_term = sides_transformed * 2

    distances = torch.norm(
        coords.unsqueeze(1) - centers_transformed.unsqueeze(-1).unsqueeze(-1), dim=0
    )
    neighbor_distance, _ = torch.topk(
        torch.nn.functional.relu(distances - sides_transformed) / normalization_term,
        k=num_neighbor,
        dim=0,
        largest=False,
    )

    regularization_term = coe_joint * torch.sum(neighbor_distance**2)

    return regularization_term


def cal_loss(
    image,
    shapes,
    shape_groups,
    clip_model,
    text_features,
    coe_dict,
    use_aug=True,
    augment_trans=None,
    use_neg=True,
    text_features_neg=None,
    verbose=True,
):
    # Transform image for CLIP input
    image = image[:, :, 3:4] * image[:, :, :3] + torch.ones(
        image.shape[0], image.shape[1], 3, device=pydiffvg.get_device()
    ) * (1 - image[:, :, 3:4])
    image = image[:, :, :3]
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Compute the loss
    pos_clip_loss = torch.zeros(1, device=pydiffvg.get_device())
    neg_clip_loss = torch.zeros(1, device=pydiffvg.get_device())
    NUM_AUGS = 1
    img_augs = [image]
    if use_aug:
        NUM_AUGS = 4
        for n in range(NUM_AUGS - 1):
            img_augs.append(augment_trans(image))
    img_batch = torch.cat(img_augs)
    image_features = clip_model.encode_image(img_batch)
    for n in range(NUM_AUGS):
        pos_clip_loss -= torch.cosine_similarity(
            text_features, image_features[n : n + 1], dim=1
        )
        if use_neg:
            neg_clip_loss += (
                torch.cosine_similarity(
                    text_features_neg, image_features[n : n + 1], dim=1
                )
                * coe_dict["neg_clip_coe"]
            )

    # Regularization term
    diffvg_regularization_loss = torch.zeros(1, device=pydiffvg.get_device())
    if (
        torch.norm(coe_dict["delta_coe"]) > 0
        or torch.norm(coe_dict["displacement_coe"]) > 0
        or torch.norm(coe_dict["angle_coe"]) > 0
    ):
        diffvg_regularization_loss = diffvg_regularization_term(
            shapes,
            shape_groups,
            coe_delta=coe_dict["delta_coe"],
            coe_displacement=coe_dict["displacement_coe"],
            coe_angle=coe_dict["angle_coe"],
        )

    pairwise_diffvg_regularization_loss = torch.zeros(1, device=pydiffvg.get_device())
    if (
        torch.norm(coe_dict["overlap_coe"]) > 0
        or torch.norm(coe_dict["neighbor_coe"]) > 0
    ):
        pairwise_diffvg_regularization_loss = pairwise_diffvg_regularization_term(
            shapes,
            shape_groups,
            coe_overlap=coe_dict["overlap_coe"],
            num_neighbor=coe_dict["neighbor_num"],
            coe_neighbor=coe_dict["neighbor_coe"],
        )

    image_regularization_loss = torch.zeros(1, device=pydiffvg.get_device())
    if torch.norm(coe_dict["image_coe"]) > 0:
        image_regularization_loss = image_regularization_term(
            image, coe_image=coe_dict["image_coe"]
        )

    joint_regularization_loss = torch.zeros(1, device=pydiffvg.get_device())
    if torch.norm(coe_dict["joint_coe"]) > 0:
        joint_regularization_loss = joint_regularization_term(
            shapes, shape_groups, image, num_neighbor=1, coe_joint=coe_dict["joint_coe"]
        )

    loss = (
        pos_clip_loss
        + neg_clip_loss
        + diffvg_regularization_loss
        + pairwise_diffvg_regularization_loss
        + image_regularization_loss
        + joint_regularization_loss
    )

    if verbose:
        print("pos_clip_loss:", pos_clip_loss.item())
        print("neg_clip_loss:", neg_clip_loss.item())
        print("diffvg_regularization_loss:", diffvg_regularization_loss.item())
        print(
            "pairwise_diffvg_regularization_loss:",
            pairwise_diffvg_regularization_loss.item(),
        )
        print("image_regularization_loss:", image_regularization_loss.item())
        print("joint_regularization_loss:", joint_regularization_loss.item())
        print("loss:", loss.item())

    return loss, pos_clip_loss


# ----------------------- Post-processing -----------------------


def delete_rect_iter(
    canvas_width,
    canvas_height,
    render,
    shapes,
    shape_groups,
    clip_model,
    text_features,
    coe_dict,
    seed=0,
):
    # Early stop if the maximum margin is less than EPS
    EPS = 1e-4

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        seed + 1,  # seed
        None,  # background_image
        *scene_args
    )
    with torch.no_grad():
        _, loss_before = cal_loss(
            img,
            shapes,
            shape_groups,
            clip_model,
            text_features,
            coe_dict,
            use_aug=False,
            augment_trans=None,
            use_neg=False,
            text_features_neg=None,
            verbose=False,
        )

    loss_after = torch.zeros(1, device=pydiffvg.get_device())
    idx_delete = -1
    for idx, (rect, rect_group) in enumerate(zip(shapes, shape_groups)):
        shapes.pop(idx)
        shape_groups.pop(idx)

        # Shift shape_ids
        for i in range(idx, len(shapes)):
            shape_groups[i].shape_ids -= 1

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = render(
            canvas_width,  # width
            canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            seed + 1,  # seed
            None,  # background_image
            *scene_args
        )
        with torch.no_grad():
            _, loss_delete = cal_loss(
                img,
                shapes,
                shape_groups,
                clip_model,
                text_features,
                coe_dict,
                use_aug=False,
                augment_trans=None,
                use_neg=False,
                text_features_neg=None,
                verbose=False,
            )

        if loss_delete < min(loss_before - EPS, loss_after):
            loss_after = loss_delete
            idx_delete = idx

        # Recover original shapes and shape_groups
        for i in range(idx, len(shapes)):
            shape_groups[i].shape_ids += 1
        shapes.insert(idx, rect)
        shape_groups.insert(idx, rect_group)

    if idx_delete != -1:
        shapes.pop(idx_delete)
        shape_groups.pop(idx_delete)
        for i in range(idx_delete, len(shapes)):
            shape_groups[i].shape_ids -= 1

    return loss_before, loss_after


def postprocess_delete_rect(
    canvas_width,
    canvas_height,
    render,
    shapes,
    shape_groups,
    clip_model,
    text_features,
    verbose=True,
    iteration=-1
):
    assert len(shapes) == len(shape_groups)
    # We care only about pos_clip_loss when doing post-processing
    coe_dict = {
        "neg_clip_coe": 0.0,
        "delta_coe": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "displacement_coe": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "angle_coe": torch.tensor(0.0, dtype=torch.float32),
        "image_coe": torch.tensor(0.0, dtype=torch.float32),
        "overlap_coe": torch.tensor(0.0, dtype=torch.float32),
        "neighbor_num": 1,
        "neighbor_coe": torch.tensor(0.0, dtype=torch.float32),
        "joint_coe": torch.tensor(0.0, dtype=torch.float32),
    }

    t = 0
    while len(shapes) > 0:
        if iteration != -1 and t >= iteration:
            break
        print("Post-process(delete) iteration:", t)
        # The loss may not be strictly decreasing because the seed for rendering is not fixed
        len_before = len(shapes)
        loss_before, loss_after = delete_rect_iter(
            canvas_width,
            canvas_height,
            render,
            shapes,
            shape_groups,
            clip_model,
            text_features,
            coe_dict,
            seed=t,
        )
        len_after = len(shapes)
        if len_after == len_before:
            print("No more rectangles to be deleted. Early stop.")
            break
        if verbose:
            print("len_before:", len_before)
            print("len_after:", len_after)
            print("loss_before:", loss_before)
            print("loss_after:", loss_after)
        t += 1
