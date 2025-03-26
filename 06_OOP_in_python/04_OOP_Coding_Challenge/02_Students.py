class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def display_details(self):
        print(f"Name: {self.name}, Age: {self.age}, Grade: {self.grade}")

# Test
student = Student("John", 16, "10th")
student.display_details()  # Output: Name: John, Age: 16, Grade: 10th