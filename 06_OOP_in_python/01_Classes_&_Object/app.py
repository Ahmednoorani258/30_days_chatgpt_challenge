# 🚀 Day 6: Object-Oriented Programming (OOP) in Python
"""
Welcome to Day 6 of your Python learning journey! Today, we’re diving into Object-Oriented Programming (OOP), 
a powerful paradigm that helps you write modular, reusable, and scalable code. OOP is especially important for 
building complex applications like AI agents and generative AI systems.

### 📌 Objectives for Today:
1. **Understand Classes & Objects**: Learn how to create blueprints and instances in Python.
2. **Master Key OOP Principles**: Encapsulation, Inheritance, Polymorphism, and Abstraction.
3. **Build Real-World Projects**: Apply OOP to practical examples like a To-Do List Manager.
4. **Solve Coding Challenges**: Reinforce your skills with beginner and advanced exercises.

By the end, you’ll have a solid foundation in OOP and be ready to tackle sophisticated Python projects.
Let’s break it down step by step.
"""

# ----------------------------------------------
# 1️⃣ Classes & Objects in Python
# ----------------------------------------------

"""
### What is a Class?

A class is like a blueprint or template for creating objects. It defines the properties (attributes) and behaviors 
(methods) that objects created from it will have. Think of it as a design for a car—specifying it has a brand, model, and year.

### What is an Object?

An object is an instance of a class. It’s a specific entity built from the class blueprint. For example, a Toyota Camry 
from the car design is an object.

### Key Concepts:

1. **Instance Variables**: Attributes unique to each object (e.g., a car’s brand).

2. **Class Variables**: Attributes shared across all objects of the class (e.g., all cars have 4 wheels).

3. **`__init__` Method**: A special method (constructor) that runs when an object is created, initializing its attributes.
"""

# Example: Car Class
class Car:
    # Class variable (shared by all cars)
    wheels = 4

    # Constructor method to initialize instance variables
    def __init__(self, brand, model, year):
        self.brand = brand  # Instance variable
        self.model = model  # Instance variable
        self.year = year    # Instance variable

    # Method to display car details
    def display_details(self):
        print(f"Brand: {self.brand}, Model: {self.model}, Year: {self.year}")

# Creating objects (instances) of the Car class
car1 = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2018)

# Calling the method on objects
car1.display_details()  # Output: Brand: Toyota, Model: Camry, Year: 2020
car2.display_details()  # Output: Brand: Honda, Model: Civic, Year: 2018

# Accessing the class variable
print(f"Number of wheels (class variable): {Car.wheels}")  # Output: 4

"""
### Explanation:
1. **Class Variable**:
   - `wheels`: A class variable, shared by all Car objects (all cars have 4 wheels).
2. **Instance Variables**:
   - `self.brand`, `self.model`, `self.year`: Attributes unique to each car object.
3. **`__init__` Method**:
   - Initializes each car’s attributes when the object is created.
4. **`display_details` Method**:
   - A method that prints the car’s details. The `self` parameter refers to the object calling the method.
"""