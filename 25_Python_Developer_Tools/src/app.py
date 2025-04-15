from my_cool_project.core import dance
# 3. Debugging: Being a Code Detective 🔍
# What’s Debugging?
# Debugging is like being a detective solving a mystery. When your code doesn’t work (like a toy that’s broken), you use tools to figure out why and fix it.

# Definition: Debugging is finding and fixing mistakes (called bugs) in your code.
# What’s breakpoint()?
# breakpoint() is a magic word in Python that pauses your code so you can look around. It’s like pressing pause on a video game to see what’s going on.

# How to Use It
# Add a Breakpoint:
# Put breakpoint() in your code where you want to pause.
# Example:
# python

# Copy
# def add(a, b):
#     result = a + b
#     breakpoint()  # Pause here to check things
#     return result

# print(add(2, 3))
# Run Your Code:
# Run it with python my_script.py.
# It stops at breakpoint(), and you’ll see a special screen called pdb (Python Debugger).
# You can type commands like:
# p a: Shows the value of a.
# p result: Shows the value of result.
# n: Moves to the next line.
# c: Continues running.
# q: Quits debugging.

# def add(a, b):
#     result = a + b
#     # breakpoint()  # Pause here to check things
#     return result

# print(add(2, 3))

def dance():
    for i in range(100000000):  # Lots of steps!
        pass
dance()

# python -m cProfile robot.py  (run this command)

dance()  # Robot dances!