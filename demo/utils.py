import torch
import torchvision.transforms as transforms

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
        # delta regularization term
        regularization_term += torch.sum(coe_delta * ((shape.delta / shape.size) ** 2))

        # displacement regularization term
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

        # angle regularization term
        regularization_term += coe_angle * ((shape_group.angle / TWO_PI) ** 2)

    return regularization_term


def pairwise_diffvg_regularization_term(
    shapes,
    shape_groups,
    coe_overlap=torch.tensor(1.0),
    num_neighbor=1,
    coe_neighbor=torch.tensor(1.0),
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
    sides_transformed = torch.stack(
        [torch.mean(shape.size / 2 + shape.delta / 2) for shape in shapes], dim=-1
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
    pairwise_distance = torch.norm(
        centers_transformed.unsqueeze(-1) - rolling_centers_transformed, dim=0
    )
    pairwise_sum_sides = sides_transformed.unsqueeze(-1) + rolling_sides_transformed
    normalization_term = torch.tensor(
        [torch.mean(shape.size) for shape in shapes]
    ).unsqueeze(-1)

    regularization_term = 0

    # overlap regularization term
    regularization_term += coe_overlap * torch.sum(
        (
            torch.nn.functional.relu(pairwise_sum_sides - pairwise_distance)
            / normalization_term
        )
        ** 2
    )

    # neighbor regularization term (neighbors not too far apart)
    neighbor_distance, _ = torch.topk(
        torch.nn.functional.relu(2 * pairwise_distance - pairwise_sum_sides)
        / normalization_term,
        k=num_neighbor,
        dim=-1,
        largest=False,
    )
    regularization_term += coe_neighbor * torch.sum(neighbor_distance ** 2)

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
