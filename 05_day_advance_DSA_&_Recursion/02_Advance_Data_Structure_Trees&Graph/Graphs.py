# 📌 Introduction to Graphs: Nodes and Edges
"""
A graph is a way to represent relationships between objects. For example, in a social network:
- **Nodes (or Vertices):** Represent entities (e.g., people in the network).
- **Edges:** Represent connections between nodes (e.g., friendships).

### Types of Graphs:
1. **Undirected Graphs:** Edges have no direction (e.g., a two-way road).
2. **Directed Graphs:** Edges have a direction (e.g., a one-way street).

### Example:
For cities A, B, C, and D connected by roads:
- **Nodes:** A, B, C, D.
- **Edges:** A-B, A-C, B-D.

Graphs can be represented in two common ways:
1. **Adjacency List:** A list of neighbors for each node.
2. **Adjacency Matrix:** A 2D grid where rows and columns represent nodes, and a 1 indicates an edge.
"""

# ----------------------------------------------
# Graph Representation: Adjacency List
# ----------------------------------------------

# Example graph:
# Nodes: 0, 1, 2, 3
# Edges: 0-1, 0-2, 1-3
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0],
    3: [1]
}

"""
Adjacency List Representation:
0: [1, 2]  (Node 0 is connected to 1 and 2)
1: [0, 3]  (Node 1 is connected to 0 and 3)
2: [0]     (Node 2 is connected to 0)
3: [1]     (Node 3 is connected to 1)
"""

# ----------------------------------------------
# Graph Representation: Adjacency Matrix
# ----------------------------------------------

"""
### Adjacency Matrix:
- A 2D grid where rows and columns represent nodes.
- A 1 (or True) at position [i][j] means there’s an edge from node i to node j.
- A 0 (or False) means no edge exists.

### Example:
For the same graph (A, B, C, D with edges A-B, A-C, B-D):
  | A B C D
A | 0 1 1 0
B | 1 0 0 1
C | 1 0 0 0
D | 0 1 0 0

1 means an edge exists, 0 means no edge.
"""

# Representing the graph as an adjacency matrix
graph_matrix = [
    [0, 1, 1, 0],  # A
    [1, 0, 0, 1],  # B
    [1, 0, 0, 0],  # C
    [0, 1, 0, 0]   # D
]

"""
Why use it?
- **Good for dense graphs** (many edges) or when checking if an edge exists quickly.
- However, it uses more memory compared to an adjacency list for sparse graphs.
"""

# ----------------------------------------------
# Exercise: Representing a Graph with an Adjacency List
# ----------------------------------------------

"""
Let’s create a simple undirected graph to work with.

### Nodes:
0, 1, 2, 3 (using numbers for simplicity).

### Edges:
0-1, 0-2, 1-3 (these are the connections).

Since it’s undirected, each edge goes both ways (e.g., 0-1 means 0 is connected to 1, and 1 is connected to 0).

### Adjacency List:
0: [1, 2] (0 is connected to 1 and 2)
1: [0, 3] (1 is connected to 0 and 3)
2: [0]    (2 is connected to 0)
3: [1]    (3 is connected to 1)
"""

# Representing the graph as an adjacency list
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0],
    3: [1]
}

"""
This dictionary is our graph. The key is a node, and the value is a list of its neighbors.
"""



# ----------------------------------------------
# Depth-First Search (DFS): Recursive Implementation
# ----------------------------------------------

"""
Depth-First Search (DFS) is a graph traversal algorithm that explores as far as possible along each branch before backtracking.

### Key Points:
1. **Recursive Approach:** The function calls itself to explore deeper levels.
2. **Avoiding Cycles:** Use a `visited` set to track nodes that have already been processed.

### Steps:
1. Start at a given node.
2. Mark the node as visited.
3. Process the node (e.g., print it).
4. Recursively visit all unvisited neighbors.
"""

def dfs_recursive(graph, node, visited):
    # If the node is already visited, return to avoid cycles
    if node in visited:
        return
    
    # Mark the node as visited
    visited.add(node)
    
    # Process the node (e.g., print it)
    print(node)
    
    # Recursively visit all neighbors
    for neighbor in graph[node]:
        dfs_recursive(graph, neighbor, visited)

def dfs(graph, start):
    # Initialize an empty set to track visited nodes
    visited = set()
    # Start the recursive DFS from the given node
    dfs_recursive(graph, start, visited)

# Run DFS starting from node 0
print("DFS Traversal:")
dfs(graph, 0)

"""
### How It Works:
For the graph:
0: [1, 2]
1: [0, 3]
2: [0]
3: [1]

Starting from node 0:
1. Visit 0, mark it as visited.
2. Visit 1 (neighbor of 0), mark it as visited.
3. Visit 3 (neighbor of 1), mark it as visited.
4. Backtrack to 1, then to 0.
5. Visit 2 (neighbor of 0), mark it as visited.

### Output:
0
1
3
2
"""
"""
How It Works: A Walkthrough

Let’s see what happens when we call `dfs(graph, 0)`:

### Initial Call:
- `dfs(graph, 0)`
- `visited = set()` (empty).
- Calls `dfs_recursive(graph, 0, visited)`.

### Step-by-Step Execution:

1. **dfs_recursive(0):**
   - Node `0` is not in `visited`, so add `0` to `visited` → `visited = {0}`.
   - Print `0`.
   - Neighbors of `0` are `[1, 2]`.
   - Call `dfs_recursive(1)`.

2. **dfs_recursive(1):**
   - Node `1` is not in `visited`, so add `1` to `visited` → `visited = {0, 1}`.
   - Print `1`.
   - Neighbors of `1` are `[0, 3]`.
   - Node `0` is already in `visited`, so skip it.
   - Call `dfs_recursive(3)`.

3. **dfs_recursive(3):**
   - Node `3` is not in `visited`, so add `3` to `visited` → `visited = {0, 1, 3}`.
   - Print `3`.
   - Neighbor of `3` is `[1]`.
   - Node `1` is already in `visited`, so skip it.
   - No more neighbors, backtrack to `dfs_recursive(1)`.

4. **Back to dfs_recursive(1):**
   - No more neighbors to explore, backtrack to `dfs_recursive(0)`.

5. **Back to dfs_recursive(0):**
   - Continue with the next neighbor `2`.
   - Call `dfs_recursive(2)`.

6. **dfs_recursive(2):**
   - Node `2` is not in `visited`, so add `2` to `visited` → `visited = {0, 1, 3, 2}`.
   - Print `2`.
   - Neighbor of `2` is `[0]`.
   - Node `0` is already in `visited`, so skip it.
   - Backtrack to `dfs_recursive(0)`.

7. **Back to dfs_recursive(0):**
   - No more neighbors, function ends.

### Final Output:
The nodes are visited in the following order:
"""

# ----------------------------------------------
# Bonus: Collecting the DFS Traversal Order
# ----------------------------------------------

"""
If you want to collect the order of nodes visited during DFS, you can modify the function to append nodes to a list.
"""

def dfs_recursive_with_order(graph, node, visited, order):
    if node in visited:
        return
    visited.add(node)
    order.append(node)
    for neighbor in graph[node]:
        dfs_recursive_with_order(graph, neighbor, visited, order)

def dfs_with_order(graph, start):
    visited = set()
    order = []
    dfs_recursive_with_order(graph, start, visited, order)
    return order

# Run DFS and collect the traversal order
dfs_order = dfs_with_order(graph, 0)
print("\nDFS Traversal Order:", dfs_order)  # Output: [0, 1, 3, 2]

# ----------------------------------------------
# Summary
# ----------------------------------------------

"""
### Key Takeaways:
1. **Graphs:** Represent relationships between objects using nodes and edges.
2. **Adjacency List:** Efficient for sparse graphs, stores neighbors for each node.
3. **DFS (Recursive):** Explores as far as possible along each branch before backtracking.
4. **Avoiding Cycles:** Use a `visited` set to ensure each node is processed only once.

### Applications of DFS:
- Detecting cycles in a graph.
- Solving puzzles (e.g., mazes).
- Topological sorting (for directed acyclic graphs).
"""