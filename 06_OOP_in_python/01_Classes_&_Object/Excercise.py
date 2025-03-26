
# Excercise 1
# Create a Car class with attributes like brand, model, and year.

class Car():
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
    
    def display_details(self):
        print(f"Brand: {self.brand}, Model: {self.model}, Year: {self.year}")
        
    
# Excercise 2

# Instantiate multiple objects and print their details.
car = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2025)
car.display_details()
car2.display_details()



