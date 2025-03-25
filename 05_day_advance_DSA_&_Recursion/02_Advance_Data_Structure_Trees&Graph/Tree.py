# 📌 Trees
"""
A tree is a hierarchical data structure with nodes connected by edges. Each node has a parent (except the root) 
and zero or more children. A binary tree restricts each node to at most two children (left and right).
"""

# ----------------------------------------------
# Binary Tree Structure
# ----------------------------------------------

# Let’s define a simple binary tree in Python:

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Create the binary tree
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

"""
This creates the following binary tree:

        1
       / \
      2   3
     / \
    4   5
"""

# ----------------------------------------------
# Traversing a Tree
# ----------------------------------------------

"""
Tree traversals visit all nodes in a specific order. Here are the three main recursive traversals:

1. **Inorder Traversal (Left, Root, Right)**:
   - Visit the left subtree, then the root, then the right subtree.
2. **Preorder Traversal (Root, Left, Right)**:
   - Visit the root, then the left subtree, then the right subtree.
3. **Postorder Traversal (Left, Right, Root)**:
   - Visit the left subtree, then the right subtree, then the root.
"""

# Inorder Traversal (Left, Root, Right)
def inorder_traversal(node):
    if node:
        inorder_traversal(node.left)
        print(node.value, end=" ")
        inorder_traversal(node.right)

print("Inorder Traversal:")
inorder_traversal(root)  # Output: 4 2 5 1 3

print("\n")

# Preorder Traversal (Root, Left, Right)
def preorder_traversal(node):
    if node:
        print(node.value, end=" ")
        preorder_traversal(node.left)
        preorder_traversal(node.right)

print("Preorder Traversal:")
preorder_traversal(root)  # Output: 1 2 4 5 3

print("\n")

# Postorder Traversal (Left, Right, Root)
def postorder_traversal(node):
    if node:
        postorder_traversal(node.left)
        postorder_traversal(node.right)
        print(node.value, end=" ")

print("Postorder Traversal:")
postorder_traversal(root)  # Output: 4 5 2 3 1

# ----------------------------------------------
# Use Cases of Tree Traversals
# ----------------------------------------------

"""
1. **Inorder Traversal**:
   - Used to get nodes in sorted order (for binary search trees).
2. **Preorder Traversal**:
   - Used to copy a tree or serialize it.
3. **Postorder Traversal**:
   - Used to delete a tree (process children before the parent).
"""

# ----------------------------------------------
# Advanced Note: Binary Search Tree (BST)
# ----------------------------------------------

"""
In a Binary Search Tree (BST):
- For each node, the left child’s value is less than the node’s value.
- The right child’s value is greater than the node’s value.

Inorder traversal of a BST gives values in sorted order. While this example focuses on basic binary trees, 
BSTs are a great next step for learning advanced tree structures.
"""