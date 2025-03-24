# Program 1: Find the Most Frequent Element in a List
# ______________________________________________________

my_list = [1, 2, 3, 2, 4, 2, 5, 3]

# Using a dictionary to count occurrences
frequency = {}
for item in my_list:
    if item in frequency:
        frequency[item] += 1
    else:
        frequency[item] = 1

# Find the item with the maximum frequency
most_frequent = max(frequency, key=frequency.get)

print(f"List: {my_list}")
print(f"Most frequent element: {most_frequent} (appears {frequency[most_frequent]} times)")

# Explanation:
#     This program counts how often each element appears in the list using a dictionary. 
#     The 'max' function with 'key=frequency.get' finds the element with the highest count.

# Program 2: Convert a List into a Set to Remove Duplicates
# ______________________________________________________

original_list = ["apple", "banana", "apple", "cherry", "banana", "date"]

# Convert list to set to remove duplicates
unique_set = set(original_list)

print(f"Original list: {original_list}")
print(f"Set with duplicates removed: {unique_set}")

# Explanation:
#     A set automatically removes duplicates because it only stores unique elements. 
#     Converting the list to a set eliminates redundancy, though the order may not be preserved.

# Program 3: Merge Two Dictionaries
# ______________________________________________________

dict1 = {"name": "Obaid", "age": 19}
dict2 = {"city": "Lahore", "major": "Computer Science"}

# Merge using the '|' operator (Python 3.9+)
merged_dict = dict1 | dict2

print(f"Dictionary 1: {dict1}")
print(f"Dictionary 2: {dict2}")
print(f"Merged dictionary: {merged_dict}")

# Alternative method (works in all Python versions):
# merged_dict = {**dict1, **dict2}

# Explanation:
#     The '|' operator merges two dictionaries into one. If there are duplicate keys, 
#     the value from the second dictionary overwrites the first. The alternative unpacking 
#     method (commented) achieves the same result and works in older Python versions.