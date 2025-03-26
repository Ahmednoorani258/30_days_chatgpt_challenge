class Product:
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

class Inventory:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        self.products.append(product)
        print(f"Added {product.name}.")

    def remove_product(self, name):
        for product in self.products:
            if product.name == name:
                self.products.remove(product)
                print(f"Removed {name}.")
                return
        print("Product not found.")

    def update_quantity(self, name, quantity):
        for product in self.products:
            if product.name == name:
                product.quantity = quantity
                print(f"Updated {name} quantity to {quantity}.")
                return
        print("Product not found.")

    def display_inventory(self):
        for p in self.products:
            print(f"Name: {p.name}, Price: ${p.price}, Quantity: {p.quantity}")

# Test
inv = Inventory()
p1 = Product("Laptop", 999, 10)
p2 = Product("Phone", 499, 20)
inv.add_product(p1)
inv.add_product(p2)
inv.display_inventory()
# Output:
# Name: Laptop, Price: $999, Quantity: 10
# Name: Phone, Price: $499, Quantity: 20

inv.update_quantity("Laptop", 8)
inv.remove_product("Phone")
inv.display_inventory()
# Output:
# Name: Laptop, Price: $999, Quantity: 8