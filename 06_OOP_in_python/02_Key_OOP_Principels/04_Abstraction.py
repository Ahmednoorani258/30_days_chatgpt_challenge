# 📌 Abstraction
"""
Abstraction hides complex details and shows only what’s necessary. In Python, this is done using abstract classes 
that define methods child classes must implement.

### Key Points:
1. **Abstract Classes**:
   - Abstract classes act as blueprints for other classes.
   - They cannot be instantiated directly.
2. **Abstract Methods**:
   - Methods declared in an abstract class but without implementation.
   - Child classes must provide their own implementation for these methods.
3. **`abc` Module**:
   - Python’s `abc` (Abstract Base Class) module is used to create abstract classes and methods.

### Example: Abstract Shape Class
The `Shape` class is an abstract class with an abstract `area()` method. The `Circle` and `Square` classes inherit from `Shape` 
and provide their own implementation of the `area()` method.
"""

from abc import ABC, abstractmethod

# Abstract Base Class
class Shape(ABC):
    @abstractmethod
    def area(self):
        """Abstract method to calculate the area of a shape."""
        pass  # No implementation in the abstract class

# Concrete Class: Circle
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        """Implementation of the area method for a circle."""
        return 3.14 * self.radius ** 2

# Concrete Class: Square
class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        """Implementation of the area method for a square."""
        return self.side ** 2

# Usage Example
if __name__ == "__main__":
    # Create a list of shapes
    shapes = [Circle(5), Square(4)]

    # Calculate and print the area of each shape
    for shape in shapes:
        print(f"Area: {shape.area()}")

"""
### Output:
Area: 78.5
Area: 16

### Explanation:
1. **Shape Class (Abstract)**:
   - Defines the abstract `area()` method, which must be implemented by child classes.
   - Cannot be instantiated directly.

2. **Circle and Square Classes (Concrete)**:
   - Inherit from the `Shape` class.
   - Provide their own implementation of the `area()` method.

3. **`abc` Module**:
   - The `abc` module is used to create abstract classes and methods in Python.

### Benefits of Abstraction:
- Provides a clear blueprint for child classes.
- Ensures consistency by enforcing the implementation of required methods.
- Hides unnecessary details, focusing only on essential functionality.
"""