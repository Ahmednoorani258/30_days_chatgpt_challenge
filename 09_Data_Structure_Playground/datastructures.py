from collections import deque


class MyList:
    def __init__(self):
        self.items = []

    def append(self, value):
        self.items.append(value)
        return True

    def insert(self, index, value):
        if 0 <= index <= len(self.items):
            self.items.insert(index, value)
            return True
        return False

    def remove(self, value):
        if value in self.items:
            self.items.remove(value)
            return True
        return False

    def clear(self):
        self.items.clear()

    def copy(self):
        return self.items.copy()

    def count(self, value):
        return self.items.count(value)

    def extend(self, iterable):
        self.items.extend(iterable)

    def index(self, value):
        return self.items.index(value)

    def pop(self):
        if self.items:
            return self.items.pop()
        return None

    def reverse(self):
        self.items.reverse()

    def sort(self):
        self.items.sort()

    def get(self, index):
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def get_state(self):
        return self.items


class Stack:
    def __init__(self):
        self.items = []

    def push(self, value):
        self.items.append(value)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def get_state(self):
        return self.items[::-1]  # Reverse for top-first display


class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, value):
        self.items.append(value)

    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def get_state(self):
        return list(self.items)


class MySet:
    def __init__(self):
        self.items = set()

    def add(self, value):
        self.items.add(value)
        # Adds the element `value` to the set.

    def remove(self, value):
        if value in self.items:
            self.items.remove(value)
            return True
        return False
        # Removes the element `value` from the set if it exists.

    def contains(self, value):
        return value in self.items
        # Checks if the set contains the element `value`.

    def clear(self):
        self.items.clear()
        # Removes all elements from the set.

    def pop(self):
        if self.items:
            return self.items.pop()
        return None
        # Removes and returns an arbitrary element from the set.

    def union(self, other_set):
        return self.items.union(other_set)
        # Returns a new set with all elements from both sets.

    def intersection(self, other_set):
        return self.items.intersection(other_set)
        # Returns a new set with elements common to both sets.

    def difference(self, other_set):
        return self.items.difference(other_set)
        # Returns a new set with elements in the set but not in `other_set`.

    def symmetric_difference(self, other_set):
        return self.items.symmetric_difference(other_set)
        # Returns a new set with elements in either set but not in both.

    def is_subset(self, other_set):
        return self.items.issubset(other_set)
        # Checks if the set is a subset of `other_set`.

    def is_superset(self, other_set):
        return self.items.issuperset(other_set)
        # Checks if the set is a superset of `other_set`.

    def get_state(self):
        return list(self.items)
        # Returns the current state of the set as a list.


class MyDict:
    def __init__(self):
        self.items = {}

    def set(self, key, value):
        self.items[key] = value
        # Sets the `key` to the specified `value` in the dictionary.

    def get(self, key):
        return self.items.get(key)
        # Retrieves the value associated with the `key`.

    def remove(self, key):
        if key in self.items:
            del self.items[key]
            return True
        return False
        # Removes the `key` and its associated value from the dictionary.

    def clear(self):
        self.items.clear()
        # Removes all key-value pairs from the dictionary.

    def keys(self):
        return list(self.items.keys())
        # Returns a list of all keys in the dictionary.

    def values(self):
        return list(self.items.values())
        # Returns a list of all values in the dictionary.

    def items(self):
        return list(self.items.items())
        # Returns a list of all key-value pairs in the dictionary.

    def pop(self, key):
        if key in self.items:
            return self.items.pop(key)
        return None
        # Removes the `key` and returns its value.

    def popitem(self):
        if self.items:
            return self.items.popitem()
        return None
        # Removes and returns the last inserted key-value pair as a tuple.

    def update(self, other_dict):
        self.items.update(other_dict)
        # Updates the dictionary with key-value pairs from `other_dict`.

    def setdefault(self, key, default=None):
        return self.items.setdefault(key, default)
        # Returns the value of `key` if it exists, otherwise sets it to `default`.

    def get_state(self):
        return self.items
        # Returns the current state of the dictionary.
