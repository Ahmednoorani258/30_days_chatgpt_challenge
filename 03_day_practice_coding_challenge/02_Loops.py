# Write a function called fizz_buzz that takes four parameters:

# start: an integer representing the starting number of the range (inclusive).
# end: an integer representing the ending number of the range (inclusive).
# fizz_num: an integer representing the multiple for "Fizz".
# buzz_num: an integer representing the multiple for "Buzz".
# The function should print numbers from start to end (inclusive). However:

# For multiples of fizz_num, print "Fizz" instead of the number.
# For multiples of buzz_num, print "Buzz" instead of the number.
# For numbers that are multiples of both fizz_num and buzz_num, print "FizzBuzz".
# For all other numbers, print the number itself.
# Constraints:

# start and end are integers where start <= end.
# fizz_num and buzz_num are positive integers greater than 0.
# The function should handle cases where start is greater than 0.

# Example:
# fizz_buzz(1, 15, 3, 5)
# Expected Output:

# 1
# 2
# Fizz
# 4
# Buzz
# Fizz
# 7
# 8
# Fizz
# Buzz
# 11
# Fizz
# 13
# 14
# FizzBuzz

def fizz_buzz(start, end, fizz_num, buzz_num):
    """Prints numbers from start to end, replacing multiples with Fizz, Buzz, or FizzBuzz."""
    for num in range(start, end + 1):
        if num % fizz_num == 0 and num % buzz_num == 0:
            print("FizzBuzz")  # Multiple of both fizz_num and buzz_num
        elif num % fizz_num == 0:
            print("Fizz")     # Multiple of fizz_num only
        elif num % buzz_num == 0:
            print("Buzz")     # Multiple of buzz_num only
        else:
            print(num)        # No multiples

fizz_buzz(1, 15, 3, 5)


# u can try adding input to the function to make it more dynamic

