from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass  
    
    def print_details(self):
        print(f"ClassName: {self.__class__.__name__}, Area: {self.area()}")
        

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2
    


class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2
    
class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

shapes = [Circle(5), Square(4), Rectangle(3, 6)]

for shape in shapes:
    print(f"Area: {shape.area()}")
    shape.print_details()