# What Are Iterators?
# In Python, an iterator is an object that allows you to traverse through a sequence of values, one at a time. Iterators are used to loop over data structures like lists, tuples, dictionaries, and more, but they can also be used to generate sequences on the fly without storing them in memory.

# Key Concepts:
# Iterable: An object that can be iterated over (e.g., lists, tuples, strings). It implements the __iter__() method, which returns an iterator.
# Iterator: An object that produces the next value in a sequence when its __next__() method is called. It also implements __iter__() to return itself.
# Iterator Protocol: A set of rules that iterators must follow, consisting of the __iter__() and __next__() methods.
# Iterator Protocol:
# For an object to be an iterator, it must implement two methods:

# __iter__(): Returns the iterator object itself. This allows the iterator to be used in a for loop or with other functions that expect iterables.
# __next__(): Returns the next value in the sequence. When there are no more values to return, it raises a StopIteration exception to signal the end of the iteration.

#____________________________________________________________________ 

# Creating Iterators
# You can create your own iterators by defining a class that implements the iterator protocol. Let's start with a simple example.

# Example: A Simple Iterator
# Suppose we want to create an iterator that returns numbers from 1 to 5.

# class MyIterator():
#     def __init__(self):
#         self.num = 1
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.num > 5:
#             raise StopIteration
#         val = self.num
#         self.num += 1
#         return val
    
# iterator = MyIterator()
# for num in iterator:
#     print(num)  # Output: 1, 2, 3, 4, 5


# How It Works:
# __init__() initializes the starting value.
# __iter__() returns the iterator object itself.
# __next__() returns the current value and increments it. When the value exceeds 5, it raises StopIteration.
# Built-in Functions: iter() and next()
# Python provides two built-in functions to work with iterators:

# iter(iterable): Returns an iterator object from an iterable.
# next(iterator): Retrieves the next value from the iterator. If there are no more values, it raises StopIteration.

# _____________________________________________________________
# Iterator Chaining
# _____________________________________________________________

# # You can chain multiple iterators together to create a single iterator that yields values from each in sequence. This is useful for combining data from multiple sources.

# # Example: Chaining Iterators
# def chain(*iterables):
#     for it in iterables:
#         for element in it:
#             yield element

# # Usage
# list1 = [1, 2, 3]
# list2 = [4, 5, 6]
# for num in chain(list1, list2):
#     print(num)  # Output: 1, 2, 3, 4, 5, 6


# _____________________________________________________________
# The itertools Module
# _____________________________________________________________

# # The itertools module in Python’s standard library offers a collection of tools for working with iterators. It includes functions for creating and combining iterators in powerful ways.

# # Example: Using itertools.count()
# # itertools.count(start, step) creates an infinite iterator that counts from a starting number with a given step.

# import itertools

# counter = itertools.count(start=10, step=2)
# for _ in range(5):
#     print(next(counter))  # Output: 10, 12, 14, 16, 18


# cycle(iterable): Cycles through an iterable indefinitely.
# repeat(element, times): Repeats an element a specified number of times.
# islice(iterable, start, stop, step): Slices an iterator like a list.

# _____________________________________________________________
# Lazy Evaluation with Iterators
# _____________________________________________________________


# # One of the key advantages of iterators is lazy evaluation, meaning values are computed only when needed. This is particularly useful for handling large datasets or infinite sequences.

# # Example: Processing Large Files
# # Suppose you have a large file that doesn’t fit into memory. You can use an iterator to read it line by line:

# def read_large_file(file_path):
#     with open(file_path, 'r') as file:
#         for line in file:
#             yield line.strip()

# # Usage
# for line in read_large_file('file.txt'):
#     # Process each line individually
#     print(line)
    
#     # Benefit: Only one line is loaded into memory at a time, making it memory-efficient.

# _____________________________________________________________
# Data Pipelines
# _____________________________________________________________

# def read_data():
#     # Simulate reading data
#     for i in range(10):
#         yield i

# def filter_even(numbers):
#     for num in numbers:
#         if num % 2 == 0:
#             yield num

# def square(numbers):
#     for num in numbers:
#         yield num ** 2

# # Create a pipeline
# data = read_data()
# even_data = filter_even(data)
# squared_data = square(even_data)

# for value in squared_data:
#     print(value)  # Output: 0, 4, 16, 36, 64

# _____________________________________________________________
# Working with APIs
# _____________________________________________________________

# When dealing with paginated APIs, iterators can help fetch and process data page by page without loading everything at once.

# Example: Fetching Paginated Data

# import requests  # Assuming requests library is installed

# def fetch_pages(api_url):
#     page = 1
#     while True:
#         response = requests.get(f"{api_url}?page={page}")
#         data = response.json()
#         if not data:
#             break
#         yield data
#         page += 1

# # Usage
# for page_data in fetch_pages('https://api.example.com/data'):
#     # Process each page
#     print(page_data)











# Summary
# Iterators are objects that allow you to traverse sequences one value at a time, following the iterator protocol (__iter__() and __next__()).
# Generator functions provide a simpler way to create iterators using the yield keyword.
# Advanced concepts include iterator chaining, the itertools module, and lazy evaluation for handling large or infinite datasets.
# Practical applications range from processing large files to creating data pipelines and handling paginated APIs.