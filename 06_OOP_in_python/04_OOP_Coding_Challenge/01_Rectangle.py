class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * (self.length + self.width)

# Test
rect = Rectangle(5, 3)
print(f"Area: {rect.area()}")        # Output: Area: 15
print(f"Perimeter: {rect.perimeter()}")  # Output: Perimeter: 16