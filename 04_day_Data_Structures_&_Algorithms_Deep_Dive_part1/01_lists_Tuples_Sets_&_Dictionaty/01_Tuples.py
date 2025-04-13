# Tuples

# Immutable, ordered collections. Once created, they cannot be changed.

# Example 1: Coordinates
# ______________________________________________________

coordinates = (40.7128, -74.0060)

print(coordinates)

# Explanation:
#     Represents a latitude and longitude pair (e.g., New York City). The values are fixed, and attempting to modify it (e.g., coordinates[0] = 50) would raise an error.

# Example 2: RGB Colors
# ______________________________________________________

rgb = (255, 128, 0)

print(rgb)

# Explanation:
#     A tuple storing red, green, and blue values for a color (orange). Its immutability ensures the color code remains constant.

# Example 3: Student Record
# ______________________________________________________

student = ("Obaid", 19, "Computer Science")

print(student)

# Explanation:
#     A tuple with a name, age, and major. It’s ordered (name comes first), and since it’s immutable, the data stays consistent.
