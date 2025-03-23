# Question: Longest Common Prefix
# Problem:

# Write a function longest_common_prefix(strs) that takes a list of strings strs as input and returns the longest common prefix shared by all strings in the list. If there is no common prefix, return an empty string. The prefix must be a contiguous sequence of characters that appears at the beginning of every string in the list.

# Requirements:

# Handle edge cases like an empty list or single-element list.
# The solution should work efficiently for strings of varying lengths.

# Examples:
# longest_common_prefix(["flower", "flow", "flight"]) → "fl" (all start with "fl")
# longest_common_prefix(["dog", "racecar", "car"]) → "" (no common prefix)
# longest_common_prefix(["interspecies", "interstellar", "interstate"]) → "inter" (common prefix is "inter")
# longest_common_prefix(["throne", "throne"]) → "throne" (identical strings)
# longest_common_prefix([]) → "" (empty list)

# Plan:
# 1. If the list is empty, return an empty string.
# 2. Sort the list of strings to find the shortest string.
# 3. Iterate through the characters of the shortest string.
# 4. Check if the character is present in all the strings.
# 5. If the character is present in all the strings, add it to the prefix.
# 6. If the character is not present in all the strings, return the prefix.
# 7. If the loop completes, return the prefix.

def longest_common_prefix(strs):
    if not strs:
        return ""
    strs.sort()
    prefix = ""
    # loop through the characters of the shortest string
    for i in range(len(strs[0])):
        # check if the character is present in all the strings
        if all(strs[0][i] == s[i] for s in strs):
            prefix += strs[0][i]
        else:
            return prefix
    return prefix

# Test cases
print(longest_common_prefix(["flower", "flow", "flight"])) # "fl"
print(longest_common_prefix(["dog", "racecar", "car"])) # ""
print(longest_common_prefix(["interspecies", "interstellar", "interstate"])) # "inter"
print(longest_common_prefix(["throne", "throne"])) # "throne"


# this code is working properly but not readable. I will refactor it to make it more readable.

# Refactored code:

def longest_common_prefix(strs):
    """Find the longest common prefix among a list of strings."""
    # Handle empty list case
    if not strs:
        return ""
    
    # If only one string, it’s the prefix
    if len(strs) == 1:
        return strs[0]
    
    # Sort the list to compare the most different strings
    strs.sort()
    first = strs[0]  # Shortest prefix candidate after sorting
    last = strs[-1]  # Longest differing string after sorting
    prefix = ""
    
    # Compare characters of the first and last strings
    for i in range(len(first)):
        # If characters match, add to prefix
        
        if i < len(last) and first[i] == last[i]:
            prefix += first[i]
        else:
            # Stop at the first mismatch
            break
    
    return prefix