from functools import wraps
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
# Decorators with Arguments (Decorator Factory)
# _____________________________________________________________

# from functools import wraps
# def repeat(n):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             for _ in range(n):
#                 result = func(*args, **kwargs)
#             return result
#         return wrapper
#     return decorator

# @repeat(5)
# def say_hello():
#     print("Hello!")

# say_hello()

# _____________________________________________________________
# Chaining Decorators
# _____________________________________________________________

# def decorator1(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         print("Decorator 1")
#         return func(*args, **kwargs)
#     return wrapper

# def decorator2(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         print("Decorator 2")
#         return func(*args, **kwargs)
#     return wrapper

# @decorator1
# @decorator2
# def say_hello():
#     print("Hello!")

# say_hello()

# _____________________________________________________________
# _____________________________________________________________
# _____________________________________________________________
# Real World Examples
# _____________________________________________________________
# _____________________________________________________________
# _____________________________________________________________


# 1) Logging Decorator
# _____________________________________________________________

# def log(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
#         return func(*args, **kwargs)
#     return wrapper

# @log
# def add(a, b):
#     return a + b

# result = add(3, 5)
# print(result)

# 2) Timing Decorator
# _____________________________________________________________

# import time

# def timer(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         print(f"{func.__name__} took {end - start:.4f} seconds")
#         return result
#     return wrapper

# @timer
# def slow_function():
#     time.sleep(5)

# slow_function()

# 3) Authentication Decorator
# _____________________________________________________________

# def requires_auth(func):
#     @wraps(func)
#     def wrapper(user, *args, **kwargs):
#         if not user.get('is_authenticated'):
#             raise PermissionError("Authentication required")
#         return func(user, *args, **kwargs)
#     return wrapper

# @requires_auth
# def secret_function(user):
#     print("This is a secret function.")

# user = {'is_authenticated': True}
# secret_function(user)

# user = {'is_authenticated': False}
# try:
#     secret_function(user)
# except PermissionError as e:
#     print(e)
# _____________________________________________________________