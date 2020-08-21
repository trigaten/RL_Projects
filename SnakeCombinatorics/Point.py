""""""
class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init
    def shift(self, x, y):
        self.x += x
        self.y += y
    def set(self, x, y):
        self.x = x
        self.y = y
