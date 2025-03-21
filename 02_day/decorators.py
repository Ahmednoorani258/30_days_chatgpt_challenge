# What is a Decorator?
# A decorator is a function (or sometimes a class) that takes another function as an argument and enhances its behavior. Think of it as a wrapper that adds functionality before or after the original function executes, all while keeping the original code intact.

# Why Use Decorators?
# Decorators are incredibly useful for:

# Logging: Tracking function calls and their arguments.
# Authentication: Verifying user permissions.
# Caching: Storing results of expensive computations.
# Timing: Measuring execution time.
# Input/Output Modification: Adjusting what goes in or comes out of a function.

# _____________________________________________________________
# Decorator Syntax
# _____________________________________________________________

# def my_decorator(func):
#     def wrapper():
#         print("Something is happening before the function is called.")
#         func()
#         print("Something is happening after the function is called.")
#     return wrapper

# @my_decorator
# def say_hello():
#     print("Hello!")

# say_hello()



# _____________________________________________________________
# Decorator Example Eith Arguments
# _____________________________________________________________

# def func(*a,**k):
#     print(a)
#     print(k)
    
# func(1,2,3,4,5,6,7,8,9,10, name="Alice", age=25)
# def my_decorator(func):
#     def wrapper(*args, **kwargs):
#         print("Before the  function call.")
#         result = func(*args, **kwargs)
#         print(f"After the function call.")
#         return result
#     return wrapper

# @my_decorator
# def greet(name):
#     print(f"Hello, {name}!")

# greet("Alice")

# _____________________________________________________________
# Preserving Function Metadata
# _____________________________________________________________

# from functools import wraps

# def my_decorator(func):
#     @wraps(func) # try commenting this line and see the difference
#     def wrapper(*args, **kwargs):
#         print("Before the function call.")
#         result = func(*args, **kwargs)
#         print("After the function call.")
#         return result
#     return wrapper

# @my_decorator
# def greet(name):
#     """Greet someone by name."""
#     print(f"Hello, {name}!")

# greet("Alice")
# print(greet.__name__)  # Outputs: greet
# print(greet.__doc__)   # Outputs: Greet someone by name.

# _____________________________________________________________
# Class-based Decorators
# _____________________________________________________________

# class MyDecorator:
#     def __init__(self, func):
#         self.func = func

#     def __call__(self, *args, **kwargs):
#         print("Before the function call.")
#         result = self.func(*args, **kwargs)
#         print("After the function call.")
#         return result

# @MyDecorator
# def greet(name):
#     print(f"Hello, {name}!")

# greet("Alice")

# _____________________________________________________________
# Decorator Syntax
# _____________________________________________________________