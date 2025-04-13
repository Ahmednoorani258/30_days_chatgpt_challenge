# 📌 Polymorphism
"""
Polymorphism means “many forms.” It allows objects of different classes to be treated as instances of a common parent class,
often by overriding methods.

### Key Points:

1. **Polymorphism**:
   - The ability of different objects to respond to the same method call in different ways.

2. **Method Overriding**:
   - A child class provides a specific implementation of a method that is already defined in its parent class.

### Example: Method Overriding
The `Animal` class defines a generic `speak()` method, which is overridden by the `Dog` and `Cat` classes to provide specific behaviors.
"""


# Parent Class
class Animal:
    def speak(self):
        """Generic method to simulate an animal making a sound."""
        print("Animal is making a sound.")


# Child Class: Dog
class Dog(Animal):
    def speak(self):
        """Specific implementation for Dog."""
        print("Dog is barking.")


# Child Class: Cat
class Cat(Animal):
    def speak(self):
        """Specific implementation for Cat."""
        print("Cat is meowing.")


# Usage Example
if __name__ == "__main__":
    # Create a list of Animal objects (Dog and Cat)
    animals = [Dog(), Cat()]

    # Call the speak() method on each object
    for animal in animals:
        animal.speak()

"""
### Output:
Dog is barking.
Cat is meowing.

### Explanation:
1. **Animal Class (Parent)**:
   - Defines a generic `speak()` method.

2. **Dog and Cat Classes (Children)**:
   - Override the `speak()` method to provide specific behaviors.

3. **Polymorphism**:
   - The same method call (`speak()`) behaves differently depending on the object’s class (Dog or Cat).

### Benefits of Polymorphism:
- Promotes flexibility and reusability by allowing the same interface to be used for different types of objects.
- Simplifies code by enabling dynamic method resolution at runtime.
"""
