import math
import numpy as np


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)


class EquationBox:
    def __init__(self, topLeft=(0, 0), bottomRight=(0, 0)):
        self.topLeft = (max(int(topLeft[0]), 0), max(int(topLeft[1]), 0))
        self.bottomRight = (
            max(int(bottomRight[0]), 0), max(int(bottomRight[1]), 0))

    def scale(self, factor=(1.0, 1.0)):
        width, height = self.size()
        newWidth = width * factor[0]
        newHeight = height * factor[1]
        newX = self.topLeft[0] * factor[0]
        newY = self.topLeft[1] * factor[1]
        newX2 = self.bottomRight[0] * factor[0]
        newY2 = self.bottomRight[1] * factor[1]
        newBottomRight = (newX + newWidth, newY + newHeight)
        return EquationBox((newX, newY), (newX2, newY2))

    def rotate(self, origin, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        width, height = self.size()
        points = [
            (self.topLeft[0], self.topLeft[1]),
            (self.topLeft[0], self.topLeft[1] + height),
            (self.topLeft[0] + width, self.topLeft[1]),
            (self.topLeft[0] + width, self.topLeft[1] + height)
        ]
        rotated_points = np.array([rotate_point(
            origin, point, angle_radians) for point in points])

        x_min, y_min = np.min(rotated_points, axis=0)
        x_max, y_max = np.max(rotated_points, axis=0)

        return EquationBox((x_min, y_min), (x_max, y_max))

    def shift(self, shift):
        return EquationBox(
            (self.topLeft[0] + shift[0], self.topLeft[1] + shift[1]),
            (self.bottomRight[0] + shift[0], self.bottomRight[1] + shift[1])
        )

    def to_eq_coord(self):
        return {
            "x1": int(self.topLeft[0]),
            "y1": int(self.topLeft[1]),
            "x2": int(self.bottomRight[0]),
            "y2": int(self.bottomRight[1])
        }

    def from_eq_coord(coord):
        return EquationBox((coord['x1'], coord['y1']), (coord['x2'], coord['y2']))

    def to_array(self):
        return [self.topLeft[0], self.topLeft[1], self.bottomRight[0], self.bottomRight[1]]

    def center(self):
        return (
            (self.topLeft[0] + self.bottomRight[0]) / 2,
            (topLeft[1] + self.bottomRight[1]) / 2
        )

    def size(self) -> (int, int):
        return (self.bottomRight[0] - self.topLeft[0], self.bottomRight[1] - self.topLeft[1])

    def collision(self, rect) -> bool:
        width, height = self.size()
        rect_width, rect_height = rect.size()
        return self.topLeft[0] + width >= rect.topLeft[0] and \
            self.topLeft[0] <= rect.topLeft[0] + rect_width and \
            self.topLeft[1] + height >= rect.topLeft[1] and \
            self.topLeft[1] <= rect.topLeft[1] + rect_height

    def iou(self, rect) -> float:
        # iou is 0 if no collision
        if not self.collision(rect):
            return 0.0

        # compute area of intersection
        x_left = max(self.topLeft[0], rect.topLeft[0])
        y_top = max(self.topLeft[1], rect.topLeft[1])
        x_right = min(self.bottomRight[0], rect.bottomRight[0])
        y_bottom = min(self.bottomRight[1], rect.bottomRight[1])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = max(0, x_right - x_left + 1) * \
            max(0, y_bottom - y_top + 1)

        self_area = (self.bottomRight[0] - self.topLeft[0] + 1) * \
            (self.bottomRight[1] - self.topLeft[1] + 1)
        rect_area = (rect.bottomRight[0] - rect.topLeft[0] + 1) * \
            (rect.bottomRight[1] - rect.topLeft[1] + 1)

        iou = intersection_area / \
            float(self_area + rect_area - intersection_area)

        return iou

    def is_zero(self) -> bool:
        return self.topLeft[0] == 0 and \
            self.topLeft[1] == 0 and \
            self.bottomRight[0] == 0 and \
            self.bottomRight[1] == 0

    def __repr__(self):
        return f'EquationBox(topLeft={self.topLeft}, bottomRight={self.bottomRight}'

    def __eq__(self, obj):
        return isinstance(obj, EquationBox) and obj.topLeft == self.topLeft and obj.bottomRight == self.bottomRight
