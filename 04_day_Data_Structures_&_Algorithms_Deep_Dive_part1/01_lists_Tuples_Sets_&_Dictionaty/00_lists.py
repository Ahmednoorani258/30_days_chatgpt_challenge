# Lists


# Mutable, ordered collections of items. Items can be changed, added, or removed.


# Example 1: Grocery List
# ______________________________________________________

grocery_list = ["apples", "bananas", "milk"]

grocery_list.append("bread")

print(grocery_list)

# Explanation: 
    
#     This list stores a collection of grocery items in a specific order. You can modify it, e.g., append "bread" with grocery_list.append("bread"), making it ["apples", "bananas", "milk", "bread"].




# Example 2: Numbers
# ______________________________________________________
numbers = [10, 20, 30, 40]
numbers[1] = 25
print(numbers)

# Explanation: 
#     A list of integers in sequence. Being mutable, you could change an element, e.g., numbers[1] = 25, resulting in [10, 25, 30, 40].



# Example 3: Mixed Types

mixed_list = [1, "hello", 3.14, True]

print(mixed_list)

# Explanation:
#     Lists can hold different data types (integers, strings, floats, booleans). The order is preserved, and you can access elements by index, e.g., mixed_list[1] returns "hello".