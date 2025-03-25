# 📌 Tower of Hanoi
"""
The Tower of Hanoi is a classic problem in recursion. The goal is to move `n` disks from the source peg (A) 
to the target peg (C) using an auxiliary peg (B), following these rules:

### Rules:
1. Only one disk can be moved at a time.
2. A larger disk cannot be placed on top of a smaller disk.
3. All disks must be moved from the source peg to the target peg.

### Key Points:
1. **Base Case:** If `n == 1`, move the disk directly from the source peg to the target peg.
2. **Recursive Case:** 
   - Move `n-1` disks from the source peg to the auxiliary peg.
   - Move the nth disk from the source peg to the target peg.
   - Move the `n-1` disks from the auxiliary peg to the target peg.

### Example:
For `n = 3` disks:
1. Move 2 disks from A to B (using C as auxiliary).
2. Move the 3rd disk from A to C.
3. Move 2 disks from B to C (using A as auxiliary).

The steps are printed as instructions for each move.
"""

def tower_of_hanoi(n, source, auxiliary, target):
    # Base Case: If there's only one disk, move it directly
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    
    # Recursive Case:
    # Step 1: Move n-1 disks from source to auxiliary (using target as auxiliary)
    tower_of_hanoi(n - 1, source, target, auxiliary)
    
    # Step 2: Move the nth disk from source to target
    print(f"Move disk {n} from {source} to {target}")
    
    # Step 3: Move the n-1 disks from auxiliary to target (using source as auxiliary)
    tower_of_hanoi(n - 1, auxiliary, source, target)

# Example usage
print("Steps to solve Tower of Hanoi for 3 disks:")
tower_of_hanoi(3, 'A', 'B', 'C')