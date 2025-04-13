# Dictionaries

# Collections of key-value pairs. Keys are unique, and values can be accessed via keys.

# Example 1: Person Info
# ______________________________________________________

person = {"name": "Obaid", "age": 19, "city": "Lahore"}

print(person["name"])
print(person["age"])
print(person)

# Explanation:
#     Keys ("name", "age", "city") map to values ("Obaid", 19, "Lahore"). Access values like person["name"] to get "Obaid".

# Example 2: Prices
# ______________________________________________________

prices = {"apple": 0.5, "banana": 0.25, "orange": 0.75}

print(prices)

# Explanation:
#     Maps fruit names to their prices. You can update a value, e.g., prices["apple"] = 0.6, or add a new pair like "grape": 1.0.

# Example 3: Grades
# ______________________________________________________

grades = {"Math": "A", "Science": "B+", "History": "A-"}

print(grades)

# Explanation:
#     Tracks subjects and grades. Keys are subjects, values are grades. Retrieve a grade with grades["Math"] to get "A".
