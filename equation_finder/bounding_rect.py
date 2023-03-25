import math
import numpy as np


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)


class BoundingRect:
    def __init__(self, topleft=(0, 0), width=1, height=1):
        self.x = topleft[0]
        self.y = topleft[1]
        self.width = width
        self.height = height

    def from_coords(topleft=(0, 0), bottomright=(1, 1)):
        return BoundingRect(topleft, bottomright[0] - topleft[0], bottomright[1] - topleft[1])

    def to_coords(self):
        return ((self.x, self.y), (self.x + self.width, self.y + self.height))

    def scale(self, factor=1.0):
        newWidth = self.width * factor
        newHeight = self.height * factor
        newX = self.x + newWidth
        newY = self.y + newHeight
        return BoundingRect((self.x, self.y), newWidth, newHeight)

    def shift(self, shift=(0, 0)):
        newWidth = self.width
        newHeight = self.height
        newX = self.x + shift[0]
        newY = self.y + shift[1]
        return BoundingRect((newX, newY), newWidth, newHeight)

    def rotate(self, origin, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        points = [
            (self.x, self.y),
            (self.x, self.y + self.height),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height)
        ]
        rotated_points = np.array([rotate_point(
            origin, point, angle_radians) for point in points])

        x_min, y_min = np.min(rotated_points, axis=0)
        x_max, y_max = np.max(rotated_points, axis=0)

        # new_top_left = rotate_point(origin, (self.x, self.y), angle_radians)
        # new_bottom_right = rotate_point(
        #     origin, (self.x + self.width, self.y + self.height), angle_radians)
        # rotated = BoundingRect.from_coords(new_top_left, new_bottom_right)
        # min_top_left = tuple(
        #     np.min(np.array([(self.x, self.y), new_top_left]), axis=0))
        # max_bottom_right = tuple(np.max(
        #     np.array([(self.x + self.width, self.y + self.height), new_bottom_right]), axis=0))
        return BoundingRect.from_coords((x_min, y_min), (x_max, y_max))

    def to_eq_coord(self):
        return {
            "x1": self.x,
            "y1": self.y,
            "x2": self.x + self.width,
            "y2": self.y + self.height
        }

    def center(self):
        topLeft, bottomRight = self.to_coords()
        return (
            (topLeft[0] + bottomRight[0]) / 2,
            (topLeft[1] + bottomRight[1]) / 2
        )

    def collision(self, rect):
        return self.x + self.width >= rect.x and self.x <= rect.x + rect.width and self.y + self.height >= rect.y and self.y <= rect.y + rect.height
