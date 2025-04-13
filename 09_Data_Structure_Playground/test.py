# List operations
lst = []  # Initialize an empty list

lst.append(1)  # Adds the element `1` to the end of the list
lst.insert(0, 1)  # Inserts `1` at index `0`
lst.remove(1)  # Removes the first occurrence of `1` in the list
lst.clear()  # Removes all elements from the list
lst.copy()  # Returns a shallow copy of the list
lst.count(1)  # Counts the number of occurrences of `1` in the list
lst.extend([1, 2, 3])  # Extends the list by appending elements from another iterable
lst.index(1)  # Returns the index of the first occurrence of `1` in the list
lst.pop()  # Removes and returns the last element of the list
lst.reverse()  # Reverses the elements of the list in place
lst.sort()  # Sorts the list in ascending order

# Dictionary operations
dic = {}  # Initialize an empty dictionary

dic.clear()  # Removes all items from the dictionary
dic.copy()  # Returns a shallow copy of the dictionary
dic.fromkeys(
    [1, 2, 3]
)  # Creates a new dictionary with keys from the iterable and values set to `None`
dic.get(1)  # Returns the value for the key `1` if it exists, otherwise returns `None`
dic.items()  # Returns a view object of the dictionary's key-value pairs
dic.keys()  # Returns a view object of the dictionary's keys
dic.pop(1)  # Removes the key `1` and returns its value
dic.popitem()  # Removes and returns the last inserted key-value pair as a tuple
dic.setdefault(
    1
)  # Returns the value of key `1` if it exists, otherwise sets it to `None` and returns it
dic.update(
    {1: 2}
)  # Updates the dictionary with the key-value pairs from another dictionary or iterable
dic.values()  # Returns a view object of the dictionary's values

# Set operations
st = set()  # Initialize an empty set

st.add(1)  # Adds the element `1` to the set
st.clear()  # Removes all elements from the set
st.copy()  # Returns a shallow copy of the set
st.difference()  # Returns a new set with elements in the set that are not in the other set(s)
st.difference_update()  # Removes elements from the set that are also in the other set(s)
st.discard(
    1
)  # Removes the element `1` from the set if it exists (does not raise an error if it doesn't exist)
st.intersection()  # Returns a new set with elements common to the set and the other set(s)
st.intersection_update()  # Updates the set with elements common to the set and the other set(s)
st.isdisjoint()  # Returns `True` if the set has no elements in common with the other set
st.issubset()  # Returns `True` if the set is a subset of the other set
st.issuperset()  # Returns `True` if the set is a superset of the other set
st.pop()  # Removes and returns an arbitrary element from the set (raises an error if the set is empty)
st.remove(
    1
)  # Removes the element `1` from the set (raises an error if the element doesn't exist)
st.symmetric_difference()  # Returns a new set with elements in either the set or the other set(s) but not both
st.symmetric_difference_update()  # Updates the set with elements in either the set or the other set(s) but not both
st.union()  # Returns a new set with all elements from the set and the other set(s)
st.update()  # Updates the set with elements from the other set(s)
