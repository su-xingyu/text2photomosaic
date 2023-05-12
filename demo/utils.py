import torch
import torchvision.transforms as transforms
import pydiffvg

TWO_PI = 2 * torch.pi


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

def calc_loss(shapes, shape_groups, augment_trans, model, text_features, with_reg=False, canvas_width=224, canvas_height=224):
    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        99999, # seed
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
            shapes, shape_groups
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

def render_scene(name, shapes, shape_groups, canvas_width=224, canvas_height=224, gamma=1.0):
    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        99999,  # seed
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
