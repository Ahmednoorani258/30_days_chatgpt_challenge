# 📌 Inheritance
"""
Inheritance allows a class (child) to inherit attributes and methods from another class (parent), promoting code reuse.

### Key Points:

1. **Parent Class**: The class whose attributes and methods are inherited.
2. **Child Class**: The class that inherits from the parent class and can add its own attributes and methods.
3. **`super()`**: Used to call the parent class’s methods or constructor.

### Example: Employee Inherits from Person
The `Employee` class inherits from the `Person` class, reusing its attributes and methods while adding its own.
"""

# Parent Class
class Person:
    def __init__(self, name, age):
        self.name = name  # Public attribute
        self.age = age    # Public attribute

    def speak(self):
        """Method to simulate speaking."""
        print(f"{self.name} is speaking.")

# Child Class
class Employee(Person):
    def __init__(self, name, age, job_title):
        # Call the parent class's constructor to initialize name and age
        super().__init__(name, age)
        self.job_title = job_title  # New attribute specific to Employee

    def work(self):
        """Method to simulate working."""
        print(f"{self.name} is working as a {self.job_title}.")

# Usage Example
if __name__ == "__main__":
    # Create an Employee object
    employee = Employee("Alice", 30, "Software Engineer")

    # Call methods from both the parent and child classes
    employee.speak()  # Output: Alice is speaking.
    employee.work()   # Output: Alice is working as a Software Engineer.

"""
### Explanation:
1. **Person Class (Parent)**:
   - Attributes: `name`, `age`.
   - Method: `speak()`.

2. **Employee Class (Child)**:
   - Inherits `name`, `age`, and `speak()` from the `Person` class.
   - Adds a new attribute `job_title` and a new method `work()`.

3. **`super()`**:
   - Used in the `Employee` class to call the `__init__` method of the `Person` class, ensuring `name` and `age` are initialized.

### Benefits of Inheritance:
- Promotes code reuse by allowing child classes to inherit common functionality from parent classes.
- Enables the addition of specialized functionality in child classes without modifying the parent class.
"""