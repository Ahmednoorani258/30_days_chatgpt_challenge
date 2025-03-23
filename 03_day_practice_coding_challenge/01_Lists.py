# find the most common element. in the list

def most_common(lst):
    return max(set(lst), key=lst.count)

print(most_common([1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 5, 5])) # 5
print(most_common([1, 2, 3, 4, 5, 6, 7, 8, 9])) # 1
print(most_common([2, 2, 4, 4,])) # 1
print(most_common([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9])) # 9
# test with str
print(most_common(['a', 'b', 'c', 'd', 'e', 'e', 'e', 'f', 'g', 'h', 'i', 'e', 'e'])) # e
print(most_common(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])) # a


# Explanation
# in this function we use predefined functions like set() and max() to find the most common element in the list.
# set() function is used to remove the duplicate elements from the list.
# max() function is used to find the maximum element in the list.   
# key=lst.count is used to count the number of times the element is repeated in the list.
# the element which is repeated the most number of times is the most common element in the list.
# but there is a problem with this function. if there are two elements which are repeated the same number of times then the function will return the first element which is repeated the most number of times.
# for example, if the list is [2, 2, 4, 4] then the function will return 2 but the correct answer is 2 and 4 both are repeated the same number of times.
# so this function is not perfect but it will work in most cases.


# Now writing another custom function to find the most common element in the list.

def most_common_element(lst):
    # create a dictionary to store the count of each element in the list.
    numbers_count = {}
    # iterate through the list and count the number of times each element is repeated.
    for num in lst:
        # if the element is already in the dictionary then increment the count by 1.
        if num in numbers_count:
            numbers_count[num] += 1
        # if the element is not in the dictionary then add the element to the dictionary and set the count to 1.
        else:
            numbers_count[num] = 1
    # find the maximum count in the dictionary.
    max_count = max(numbers_count.values())
    # create a list to store the most common elements.
    results = []
    # Iterate through the dictionary and find the element which is repeated the most number of times.
    for key,value in numbers_count.items():
        # if the value is equal to the maximum count then return the key.
        if value == max_count:
            # append the key to the list
            results.append(key)
    return results
        
print(most_common_element([1, 2, 2, 3, 2]))  # Output: 2
print(most_common_element([1, 1, 2, 2]))     # Output: 1 or 2
print(most_common_element(['a', 'b', 'a']))  # Output: 'a'







# now this function works properly but the issue is that code is too long for this simple task.

def most_common_elements(lst):
    numbers_count = {}
    for num in lst:
        numbers_count[num] = numbers_count.get(num, 0) + 1
    max_count = max(numbers_count.values())
    result = [key for key, value in numbers_count.items() if value == max_count]
    return result

# now this function is shorter than the previous function and it works properly.
print(most_common_elements([1, 2, 2, 3, 2]))  # Output: 2
print(most_common_elements([1, 1, 2, 2]))     # Output: 1 or 2
print(most_common_elements(['a', 'b', 'a']))  # Output: 'a'



# so in this case we apply all steps to solve
# this problem and we have solved this problem using two different approaches.
# the first approach is to use the predefined functions to solve the problem.
# the second approach is to write a custom function to solve the problem.
# the second approach is better because it is more accurate and it works in all cases.
# the first approach is not accurate because it will return the first element which is repeated the most number of times.
# so the second approach is better than the first approach.