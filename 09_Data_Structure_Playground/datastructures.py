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
    
    def remove(self, value):
        if value in self.items:
            self.items.remove(value)
            return True
        return False
    
    def contains(self, value):
        return value in self.items
    
    def get_state(self):
        return list(self.items)

class MyDict:
    def __init__(self):
        self.items = {}
    
    def set(self, key, value):
        self.items[key] = value
    
    def get(self, key):
        return self.items.get(key)  # Returns None if key not found
    
    def remove(self, key):
        if key in self.items:
            del self.items[key]
            return True
        return False
    
    def get_state(self):
        return self.items