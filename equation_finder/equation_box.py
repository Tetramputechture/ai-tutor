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
        self.topLeft = topLeft
        self.bottomRight = bottomRight

    def scale(self, factor=1.0):
        width, height = self.size()
        newWidth = width * factor
        newHeight = height * factor
        newBottomRight = (self.topLeft[0] +
                          newWidth, self.topLeft[1] + newHeight)
        return EquationBox((self.topLeft[0], self.topLeft[1]), newBottomRight)

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

    def size(self):
        return (self.bottomRight[0] - self.topLeft[0], self.bottomRight[1] - self.topLeft[1])

    def collision(self, rect):
        width, height = self.size()
        rect_width, rect_height = rect.size()
        return self.topLeft[0] + width >= rect.topLeft[0] and \
            self.topLeft[0] <= rect.topLeft[0] + rect_width and \
            self.topLeft[1] + height >= rect.topLeft[1] and \
            self.topLeft[1] <= rect.topLeft[1] + rect_height
