# What Are Generators?
# A generator is a special type of iterator that produces values one at a time, rather than storing them all in memory at once (like lists or tuples do). This makes generators memory-efficient, especially when working with large or infinite sequences.

# Key Concept: Lazy Evaluation
# Lazy evaluation means that a generator computes its next value only when it’s requested, rather than computing all values upfront. This saves memory and can improve performance by avoiding unnecessary computations.
# Generators follow the iterator protocol, meaning they provide a __next__() method to retrieve the next value. When no more values are available, they raise a StopIteration exception.

# Creating Generators: The yield Keyword
# The most common way to create a generator is by defining a generator function, which uses the yield keyword instead of return. When you call a generator function, it doesn’t execute immediately—it returns a generator object. The function’s code runs only when you request values from the generator (e.g., using next() or a loop).

# Basic Example: Generating Squares
# Here’s a simple generator that yields the squares of numbers up to n:


# def squares(n):
#     for i in range(n):
#         yield i ** 2

# gen = squares(5)
# print(next(gen))  # Output: 0
# print(next(gen))  # Output: 1  
# print(next(gen))  # Output: 4
# print(next(gen))  # Output: 9
# print(next(gen))  # Output: 16
# # print(next(gen))  # Raises StopIteration

# # How It Works:
# # Calling squares(5) doesn’t compute anything yet—it returns a generator object.
# # Each next(gen) call runs the function until it hits a yield, yields the value, and pauses.
# # The generator resumes from where it left off on the next call.
# # Lazy Evaluation in Action: Unlike a list comprehension ([i ** 2 for i in range(5)]), which computes and stores all squares (0, 1, 4, 9, 16) in memory, the generator computes one square at a time.

# _____________________________________________________________
# Real-World Example
# _____________________________________________________________

# def read_log(file_path):
#     with open(file_path, 'r') as file:
#         for line in file:
#             yield line.strip()

# # Usage
# for line in read_log('file.txt'):
#     # Process each line (e.g., count errors, extract data)
#     print(line)
    

# _____________________________________________________________
# Generator Expressions
# _____________________________________________________________
    

# For simple generators, you can use generator expressions, which are like list comprehensions but use parentheses () instead of square brackets []. They’re concise and also lazy.
    
# evens = (i for i in range(10) if i % 2 == 0)
# print(next(evens))  # Output: 0
# print(next(evens))  # Output: 2
# print(next(evens))  # Output: 4
# # Continues: 6, 8


# squares = (x**2 for x in range(5) )
# print(next(squares))  # Output: 0
# print(next(squares))  # Output: 1    
# print(next(squares))  # Output: 4
# print(next(squares))  # Output: 9
# print(next(squares))  # Output: 16


# Benefit: Only one line is loaded into memory at a time, making it efficient for large files.

# _____________________________________________________________
# Infinite Generators
# _____________________________________________________________

# # Generators can produce infinite sequences because they generate values on demand, rather than storing them. This is impossible with lists due to memory constraints.

# # Example: Infinite Prime Numbers
# # Here’s a generator that yields prime numbers endlessly:

# def primes():
#     yield 2
#     primes_found = [2]
#     num = 3
    
#     while True:
#         if all(num % p != 0 for p in primes_found):
#             yield num
#             primes_found.append(num)
#         num += 2

# # Usage
# prime_gen = primes()
# print(next(prime_gen))  # Output: 2
# print(next(prime_gen))  # Output: 3
# print(next(prime_gen))  # Output: 5

# for p in prime_gen:
#     if p > 100:
#         break
#     print(p)

# # # Continues indefinitely

# # How It Works: The generator keeps track of previously found primes and checks each odd number to see if it’s prime, yielding it if so.
# # Stopping: Use it in a loop with a condition, e.g., for p in prime_gen: if p > 100: break.


                        # (Advanced)
# _____________________________________________________________
# 1. yield from: Delegating to Sub-Generators
# _____________________________________________________________

# # The yield from syntax (introduced in Python 3.5) lets a generator delegate yielding to another generator, simplifying nested iteration.

# # Example: Chaining Generators

# def chain(*gens):
#     for gen in gens:
#         yield from gen

# gen1 = (i for i in range(3))      # Yields: 0, 1, 2
# gen2 = (i for i in range(3, 6))   # Yields: 3, 4, 5
# for value in chain(gen1, gen2):
#     print(value)  # Output: 0, 1, 2, 3, 4, 5
    
# # Benefit: Without yield from, you’d need a nested loop (for x in gen: yield x), but yield from makes it cleaner.


# _____________________________________________________________
# 2. Two-Way Communication with send()
# _____________________________________________________________

# # Generators can receive values via the send() method, enabling two-way communication. This is useful for advanced scenarios like coroutines.

# def counter():
#     total = 0
#     while True:
#         increment = yield total
#         if increment is not None:
#             total += increment

# # Usage
# c = counter()
# print(next(c))       # Output: 0 (initializes the generator)
# print(c.send(5))     # Output: 5
# print(c.send(10))    # Output: 15

# # next(c) starts the generator, running it until the first yield and returning total (0).
# # c.send(5) sends 5 to the generator, which assigns it to increment, updates total, and yields the new total (5).
# # The process continues with each send().

# _____________________________________________________________
# 2. Two-Way Communication with send()
# _____________________________________________________________


# # 1. Processing Large Datasets in Machine Learning
# # When training a machine learning model on a dataset too large to fit in memory, a generator can yield batches of data:

# def batch_generator(data, batch_size):
#     for i in range(0, len(data), batch_size):
#         yield data[i:i + batch_size]

# # Usage
# large_dataset = list(range(1000))  # Simulate a large dataset
# for batch in batch_generator(large_dataset, 100):
#     # Train model on batch (e.g., 0-99, 100-199, etc.)
#     print(f"Processing batch: {batch[:5]}...")  # Show first 5 items


# _____________________________________________________________
# 3. Generating Sequences: Fibonacci Numbers
# _____________________________________________________________


# def fibonacci():
#     a, b = 0, 1
#     while True:
#         yield a
#         a, b = b, a + b

# # Usage
# fib = fibonacci()
# for _ in range(10):
#     print(next(fib))  # Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34


# _____________________________________________________________
# 4. Stream Processing: Real-Time Data
# _____________________________________________________________


# Generators are great for processing streams, such as sensor data or API responses:

def data_stream():
    # Simulate a stream of incoming data
    for i in range(100):
        yield i

# Usage
for data in data_stream():
    # Process each data point as it "arrives"
    print(f"Processing {data}")
    
    
    
# Summary
# Generators are lazy iterators that produce values on demand, saving memory and enabling efficient iteration.
# Creation: Use generator functions with yield or generator expressions (...).
# Features:
# Infinite sequences are possible because values are computed only when needed.
# Advanced tools like yield from simplify delegation, and send() enables two-way communication.
# Real-World Uses: Processing large files, generating sequences, handling streams, and more.