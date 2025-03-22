

# to check the output of specific example pls uncomment the code and run it
# _____________________________________________________________


# What Are Context Managers?
# A context manager in Python is a mechanism that allows you to allocate and release resources precisely when you want to, ensuring proper setup and cleanup even if errors occur. The most common use case is handling files—opening them for use and ensuring they’re closed afterward—but context managers are versatile and can manage any resource or state.

# The primary tool for using context managers is the with statement, which simplifies resource management by automating cleanup.

# Basic Example: File Handling
# Here’s a classic example of a context manager in action:

# with open('file.txt', 'r') as f:
#     data = f.read()
#     print(data)
    # Work with the file
# File is automatically closed here, even if an error occurs

# open('file.txt', 'r') is a context manager that opens the file.
# as f assigns the file object to the variable f.
# When the with block ends (or an exception occurs), the file is automatically closed.
# This ensures the file is properly closed, preventing resource leaks, without you needing to call f.close() explicitly.


# Creating Context Managers
# Python provides two primary ways to define custom context managers:

# Class-based context managers (using __enter__ and __exit__ methods).
# Generator-based context managers (using the @contextmanager decorator).
# 1. Class-Based Context Managers
# To create a context manager with a class, define a class with two special methods:

# __enter__: Called when entering the with block; it sets up the resource and returns it.
# __exit__: Called when exiting the with block; it cleans up the resource and can handle exceptions.


# _____________________________________________________________
# 1. Class-Based Context Managers
# _____________________________________________________________

# class MyContextManager:
#     def __enter__(self):
#         print("Entering the context")
#         return "Resource"  # The resource to use in the with block

#     def __exit__(self, exc_type, exc_value, traceback):
#         print("Exiting the context")
#         if exc_type:  # Check if an exception occurred
#             print(f"An exception occurred: {exc_value}")
#         # Return False (default) to propagate exceptions, or True to suppress them

# # Using the context manager
# with MyContextManager() as resource:
#     print(f"Using the resource: {resource}")
#     # Uncomment to test exception handling:
#     raise ValueError("Something went wrong")

# _____________________________________________________________
# 2. Generator-Based Context Managers
# _____________________________________________________________

# from contextlib import contextmanager

# @contextmanager
# def my_context_manager():
#     print("Entering the context")
#     try:
#         yield "Resource"  # Provide the resource to the with block
#     except Exception as e:
#         print(f"An exception occurred: {e}")
#     finally:
#         print("Exiting the context")

# # Using the context manager
# with my_context_manager() as resource:
#     print(f"Using the resource: {resource}")
#     # Uncomment to test exception handling:
#     raise ValueError("Something went wrong")


# Comparing the Two Approaches


# Feature	            Class-Based	                        Generator-Based
# Complexity	        More verbose, explicit control	    Simpler, less code
# Exception Handling	Via __exit__ arguments	            Via try/except around yield
# Suppress Exceptions	Return True in __exit__	            Handle in except block
# Use Case	            Complex logic, state management	    Simple setup/teardown


# _____________________________________________________________
# 2. Advnace (Nested Context Managers)
# _____________________________________________________________

# with open('file.txt', 'r') as f1, open('file.txt', 'r') as f2:
#     data1 = f1.read()
#     data2 = f2.read()
#     print(data1)
#     print(data2)
#     # Work with both files
    

# # For dynamic numbers of context managers, use contextlib.ExitStack:
# from contextlib import ExitStack

# files = ['file.txt', 'file.txt', 'file.txt']
# with ExitStack() as stack:
#     file_objects = [stack.enter_context(open(f, 'r')) for f in files]
    
#     # Use file_objects as a list of open files
#     # ExitStack ensures all contexts are properly exited, even if exceptions occur.
    
# _____________________________________________________________
# 2. Advnace (Temporary State Changes)
# _____________________________________________________________

# #Context managers excel at temporary changes, like switching directories:

# import os
# from contextlib import contextmanager

# @contextmanager
# def change_dir(new_dir):
#     old_dir = os.getcwd()
#     os.chdir(new_dir)
#     try:
#         yield
#     finally:
#         os.chdir(old_dir)

# # Usage
# print(f"Original dir: {os.getcwd()}")
# with change_dir('/tmp'):
#     print(f"New dir: {os.getcwd()}")
# print(f"Back to: {os.getcwd()}")

# # Original dir: /home/user
# # New dir: /tmp
# # Back to: /home/user

# _____________________________________________________________
# 2. Advnace (Context Managers with Decorators)
# _____________________________________________________________


# from contextlib import contextmanager

# @contextmanager
# def some_context():
#     print("Setup")
#     yield
#     print("Teardown")

# def apply_context(func):
#     def wrapper(*args, **kwargs):
#         with some_context():
#             return func(*args, **kwargs)
#     return wrapper

# @apply_context
# def my_function():
#     print("Function is running")

# my_function()