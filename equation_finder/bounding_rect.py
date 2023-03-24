import math


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
        newTopLeft = rotate_point(origin, (self.x, self.y), angle_radians)
        newBottomRight = rotate_point(
            origin, (self.x + self.width, self.y + self.height), angle_radians)
        return BoundingRect.from_coords(newTopLeft, newBottomRight)

    def to_eq_coord(self):
        return {
            "x1": self.x,
            "y1": self.y,
            "x2": self.x + self.width,
            "y2": self.y + self.height
        }

    def collision(self, rect):
        return self.x + self.width >= rect.x and \
            self.x <= rect.x + rect.width and \
            self.y + self.height >= rect.y and \
            self.y <= rect.y + rect.height
