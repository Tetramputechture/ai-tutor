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
        return BoundingRect((newX, newY), newWidth, newHeight)

    def shift(self, shift=(0, 0)):
        newWidth = self.width
        newHeight = self.height
        newX = self.x + shift[0]
        newY = self.y + shift[1]
        return BoundingRect((newX, newY), newWidth, newHeight)

    def collision(self, rect):
        return self.x < rect.x + rect.width \
            and self.x + self.width > rect.x \
            and self.y < rect.y + rect.height \
            and self.height + self.y > rect.y
