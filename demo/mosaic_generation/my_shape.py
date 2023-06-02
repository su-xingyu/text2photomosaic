import pydiffvg
import torch


class PolygonRect(pydiffvg.Polygon):
    """ A rectangle defined by its upper left corner, width and height.
    Vars:
        self.upper_left: upper left corner of the rectangle
        self.size: width and height of the rectangle
        self.raw_points: the four vertices of the rectangle
        self.delta: the change in width and height
        self.coe_delta: the changing rate of delta
    """
    def __init__(
        self,
        upper_left,
        width,
        height,
        coe_delta=torch.tensor([1.0, 1.0]),
        stroke_width=torch.tensor(1.0),
        id="",
    ):
        self.upper_left = upper_left
        self.size = torch.tensor([width, height])
        self.raw_points = torch.tensor(
            [
                [upper_left[0], upper_left[1]],
                [upper_left[0] + width, upper_left[1]],
                [upper_left[0] + width, upper_left[1] + height],
                [upper_left[0], upper_left[1] + height],
            ]
        )
        self.delta = torch.tensor([0.0, 0.0], requires_grad=True)
        self.coe_delta = coe_delta

        super().__init__(self.raw_points, True, stroke_width, id)

    def update(self):
        """ Update the rectangle by adding delta to the width and height.
        """
        stacked_delta = torch.stack(
            [
                torch.tensor(0.0),
                torch.tensor(0.0),
                self.coe_delta[0] * self.delta[0],
                torch.tensor(0.0),
                self.coe_delta[0] * self.delta[0],
                self.coe_delta[1] * self.delta[1],
                torch.tensor(0.0),
                self.coe_delta[1] * self.delta[1],
            ]
        ).reshape(4, 2)
        self.points = self.raw_points + stacked_delta


class RotationalShapeGroup(pydiffvg.ShapeGroup):
    """ A shape group that can be rotated and translated.
    Vars:
        self.angle: the angle of rotation
        self.translation: the translation vector
        self.coe_ang: the changing rate of angle
        self.coe_trans: the changing rate of translation
        self.color: the color of the shape group
        self._tranparent: whether the shape group is transparent
    """
    def __init__(
        self,
        shape_ids,
        fill_color,
        transparent=True,
        coe_ang=torch.tensor(1.0),
        coe_trans=torch.tensor([1.0, 1.0]),
        use_even_odd_rule=True,
        stroke_color=None,
        shape_to_canvas=torch.eye(3),
        id="",
    ):
        self.angle = torch.tensor(0.0, requires_grad=True)
        self.translation = torch.tensor([0.0, 0.0], requires_grad=True)
        self.coe_ang = coe_ang
        self.coe_trans = coe_trans
        # differential color
        self._tranparent = transparent
        if self._tranparent:
            self.color = fill_color.clone().detach().requires_grad_(True)
        else:
            self.color = fill_color[:3].clone().detach().requires_grad_(True)
        super().__init__(
            shape_ids, fill_color, use_even_odd_rule, stroke_color, shape_to_canvas, id
        )

    def update(self):
        """ Update the shape group by rotating and translating it.
        Also update the color of the shape group.
        """
        angle = self.coe_ang * self.angle
        translation = self.coe_trans * self.translation
        rotation_m = torch.stack(
            [
                torch.cos(angle),
                -torch.sin(angle),
                translation[0],
                torch.sin(angle),
                torch.cos(angle),
                translation[1],
            ]
        ).reshape(2, 3)
        self.shape_to_canvas = torch.cat(
            (rotation_m, torch.tensor([[0.0, 0.0, 1.0]])), axis=0
        )
        if self._tranparent:
            self.fill_color = self.color
        else:
            self.fill_color = torch.cat((self.color, torch.tensor([1.0])))
