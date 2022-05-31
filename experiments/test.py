class Num:
    def __init__(self):
        self.x = None

    @property
    def class_data(self):
        return self.x

    @class_data.setter
    def class_data(self, x):
        self.x = x


class Int(Num):

    def print_x(self, x):
        super(Int, self.__class__).class_data.fset(self, x)
        # super().class_data = x
        print(super(Int, self).class_data)


nn = Int()
nn.print_x(10)
