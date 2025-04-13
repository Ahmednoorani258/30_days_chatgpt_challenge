# Sets

# Unordered collections of unique elements. Duplicates are automatically removed.

# Example 1: Unique Numbers
# ______________________________________________________

unique_numbers = {1, 2, 2, 3, 3, 4}

print(unique_numbers)

# Explanation:
#     Though duplicates (2 and 3) are in the input, the set becomes {1, 2, 3, 4} because sets only store unique values. Order isn’t guaranteed.

# Example 2: Fruit Set
# ______________________________________________________

fruits = {"apple", "banana", "apple", "cherry"}

print(fruits)

# Explanation:
#     The duplicate "apple" is removed, resulting in {"apple", "banana", "cherry"}. Sets are useful for eliminating redundancy.

# Example 3: IDs
# ______________________________________________________

user_ids = {101, 102, 103, 102}
user_ids.add(104)

print(user_ids)

# Explanation:
#     Represents unique user IDs. After creation, it’s {101, 102, 103} because 102 is only stored once. You can add new IDs with .add().
