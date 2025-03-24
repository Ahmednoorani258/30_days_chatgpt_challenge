def first_non_repeating_char(s):
    char_count = {}  # Dictionary to store character frequencies
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in s:
        if char_count[char] == 1:
            return char
    return None

# Example
string = "leetcode"
result = first_non_repeating_char(string)
print(f"First non-repeating character: {result}")

string = "loveleetcode"
result = first_non_repeating_char(string)
print(f"First non-repeating character: {result}")


# Explanation:

# Use a dictionary char_count to count the frequency of each character in the string.
# Iterate through the string once to populate the dictionary.
# Iterate through the string again to find the first character with a count of 1.
# If no such character exists, return None.
# Time complexity: O(n), where n is the length of the string. Space complexity: O(k), where k is the number of unique characters.