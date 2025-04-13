import streamlit as st
from datastructures import MyList, Stack, Queue, MySet, MyDict

{
    # Define CSS for light and dark modes
    # def get_dark_mode_css():
    #     return """
    #     <style>
    #     .stApp {
    #         background-color: #1E1E1E;
    #         color: #E0E0E0;
    #     }
    #     .sidebar .sidebar-content {
    #         background-color: #2D2D2D;
    #     }
    #     .stButton>button {
    #         background-color: #4CAF50;
    #         color: white;
    #     }
    #     .stTextInput>div>input {
    #         background-color: #333;
    #         color: #E0E0E0;
    #     }
    #     .stNumberInput>div>input {
    #         background-color: #333;
    #         color: #E0E0E0;
    #     }
    #     .stSelectbox>div>div>select {
    #         background-color: #333;
    #         color: #E0E0E0;
    #     }
    #     </style>
    #     """
    # def get_light_mode_css():
    #     return """
    #     <style>
    #     .stApp {
    #         background-color: #FFFFFF;
    #         color: #000000;
    #     }
    #     .sidebar .sidebar-content {
    #         background-color: #F0F2F6;
    #     }
    #     .stButton>button {
    #         background-color: #007BFF;
    #         color: white;
    #     }
    #     .stTextInput>div>input {
    #         background-color: #FFFFFF;
    #         color: #000000;
    #     }
    #     .stNumberInput>div>input {
    #         background-color: #FFFFFF;
    #         color: #000000;
    #     }
    #     .stSelectbox>div>div>select {
    #         background-color: #FFFFFF;
    #         color: #000000;
    #     }
    #     </style>
    #     """
    # # Initialize theme state
    # if 'dark_mode' not in st.session_state:
    #     st.session_state.dark_mode = False
    # # Apply theme
    # if dark_mode:
    #     st.markdown(get_dark_mode_css(), unsafe_allow_html=True)
    # else:
    #     st.markdown(get_light_mode_css(), unsafe_allow_html=True)
    # dark_mode = st.sidebar.checkbox("Dark Mode", value=st.session_state.dark_mode)
    # st.session_state.dark_mode = dark_mode
}
# Sidebar configuration
st.sidebar.title("Settings")


# Main title
st.title("🧪 Data Structure Playground")

# Initialize data structures in session state
if "list" not in st.session_state:
    st.session_state.list = MyList()
if "stack" not in st.session_state:
    st.session_state.stack = Stack()
if "queue" not in st.session_state:
    st.session_state.queue = Queue()
if "set" not in st.session_state:
    st.session_state.set = MySet()
if "dict" not in st.session_state:
    st.session_state.dictionary = MyDict()

# Data structure selection in sidebar
data_structures = ["List", "Stack", "Queue", "Set", "Dictionary"]
selected_ds = st.sidebar.selectbox("Choose Data Structure", data_structures)

# Code snippets and complexities
list_code_snippets = {
    "append": """def append(self, value):
    self.items.append(value)
    return True
# Adds the element `value` to the end of the list.""",
    "insert": """def insert(self, index, value):
    if 0 <= index <= len(self.items):
        self.items.insert(index, value)
        return True
    return False
# Inserts the element `value` at the specified `index`.""",
    "remove": """def remove(self, value):
    if value in self.items:
        self.items.remove(value)
        return True
    return False
# Removes the first occurrence of `value` from the list.""",
    "get": """def get(self, index):
    if 0 <= index < len(self.items):
        return self.items[index]
    return None
# Retrieves the element at the specified `index`.""",
    "pop": """def pop(self):
    if self.items:
        return self.items.pop()
    return None
# Removes and returns the last element of the list.""",
    "clear": """def clear(self):
    self.items.clear()
# Removes all elements from the list.""",
    "count": """def count(self, value):
    return self.items.count(value)
# Counts the number of occurrences of `value` in the list.""",
    "reverse": """def reverse(self):
    self.items.reverse()
# Reverses the elements of the list in place.""",
    "sort": """def sort(self):
    self.items.sort()
# Sorts the list in ascending order.""",
    "extend": """def extend(self, iterable):
    self.items.extend(iterable)
# Extends the list by appending elements from another iterable.""",
    "index": """def index(self, value):
    return self.items.index(value) 
# Returns the index of the first occurrence of `value` in the list.""",
}

set_code_snippets = {
    "add": """def add(self, value):
    self.items.add(value)
# Adds the element `value` to the set.""",
    "remove": """def remove(self, value):
    if value in self.items:
        self.items.remove(value)
        return True
    return False
# Removes the element `value` from the set if it exists.""",
    "contains": """def contains(self, value):
    return value in self.items
# Checks if the set contains the element `value`.""",
    "clear": """def clear(self):
    self.items.clear()
# Removes all elements from the set.""",
    "pop": """def pop(self):
    if self.items:
        return self.items.pop()
    return None
# Removes and returns an arbitrary element from the set.""",
    "union": """def union(self, other_set):
    return self.items.union(other_set)
# Returns a new set with all elements from both sets.""",
    "intersection": """def intersection(self, other_set):
    return self.items.intersection(other_set)
# Returns a new set with elements common to both sets.""",
    "difference": """def difference(self, other_set):
    return self.items.difference(other_set)
# Returns a new set with elements in the set but not in `other_set`.""",
    "symmetric_difference": """def symmetric_difference(self, other_set):
    return self.items.symmetric_difference(other_set)
# Returns a new set with elements in either set but not in both.""",
    "is_subset": """def is_subset(self, other_set):
    return self.items.issubset(other_set)
# Checks if the set is a subset of `other_set`.""",
    "is_superset": """def is_superset(self, other_set):
    return self.items.issuperset(other_set)
# Checks if the set is a superset of `other_set`.""",
}

dict_code_snippets = {
    "set": """def set(self, key, value):
    self.items[key] = value
# Sets the `key` to the specified `value` in the dictionary.""",
    "get": """def get(self, key):
    return self.items.get(key)
# Retrieves the value associated with the `key`.""",
    "remove": """def remove(self, key):
    if key in self.items:
        del self.items[key]
        return True
    return False
# Removes the `key` and its associated value from the dictionary.""",
    "clear": """def clear(self):
    self.items.clear()
# Removes all key-value pairs from the dictionary.""",
    "keys": """def keys(self):
    return self.items.keys()
# Returns a view object of all keys in the dictionary.""",
    "values": """def values(self):
    return self.items.values()
# Returns a view object of all values in the dictionary.""",
    "items": """def items(self):
    return self.items.items()
# Returns a view object of all key-value pairs in the dictionary.""",
    "pop": """def pop(self, key):
    if key in self.items:
        return self.items.pop(key)
    return None
# Removes the `key` and returns its value.""",
    "popitem": """def popitem(self):
    if self.items:
        return self.items.popitem()
    return None
# Removes and returns the last inserted key-value pair as a tuple.""",
    "update": """def update(self, other_dict):
    self.items.update(other_dict)
# Updates the dictionary with key-value pairs from `other_dict`.""",
    "setdefault": """def setdefault(self, key, default=None):
    return self.items.setdefault(key, default)
# Returns the value of `key` if it exists, otherwise sets it to `default`.""",
}

stack_code_snippets = {
    "push": "def push(self, value):\n    self.items.append(value)",
    "pop": "def pop(self):\n    if not self.is_empty():\n        return self.items.pop()\n    return None",
    "peek": "def peek(self):\n    if not self.is_empty():\n        return self.items[-1]\n    return None",
}
queue_code_snippets = {
    "enqueue": "def enqueue(self, value):\n    self.items.append(value)",
    "dequeue": "def dequeue(self):\n    if not self.is_empty():\n        return self.items.popleft()\n    return None",
    "peek": "def peek(self):\n    if not self.is_empty():\n        return self.items[0]\n    return None",
}

list_complexity = {
    "append": "O(1)",  # Adding an element to the end of the list
    "insert": "O(n)",  # Inserting an element at a specific index
    "remove": "O(n)",  # Removing the first occurrence of an element
    "get": "O(1)",  # Accessing an element by index
    "pop": "O(1)",  # Removing the last element
    "clear": "O(n)",  # Clearing all elements from the list
    "count": "O(n)",  # Counting occurrences of an element
    "reverse": "O(n)",  # Reversing the list
    "sort": "O(n log n)",  # Sorting the list
    "extend": "O(k)",  # Extending the list with another iterable of size `k`
    "index": "O(n)",  # Finding the index of an element
}
set_complexity = {
    "add": "O(1)",  # Adding an element to the set
    "remove": "O(1)",  # Removing an element from the set
    "discard": "O(1)",  # Removing an element from the set without raising an error if it doesn't exist
    "contains": "O(1)",  # Checking if an element exists in the set
    "clear": "O(n)",  # Clearing all elements from the set
    "pop": "O(1)",  # Removing an arbitrary element
    "copy": "O(n)",  # Creating a shallow copy of the set
    "union": "O(len(s) + len(t))",  # Union of two sets
    "update": "O(len(t))",  # Adding elements from another set or iterable
    "intersection": "O(min(len(s), len(t)))",  # Intersection of two sets
    "intersection_update": "O(min(len(s), len(t)))",  # Updating the set with the intersection of itself and another
    "difference": "O(len(s))",  # Difference between two sets
    "difference_update": "O(len(t))",  # Updating the set by removing elements found in another set
    "symmetric_difference": "O(len(s))",  # Symmetric difference between two sets
    "symmetric_difference_update": "O(len(t))",  # Updating the set with the symmetric difference of itself and another
    "is_subset": "O(len(s))",  # Checking if a set is a subset of another
    "is_superset": "O(len(s))",  # Checking if a set is a superset of another
    "isdisjoint": "O(min(len(s), len(t)))",  # Checking if two sets have no elements in common
}

dict_complexity = {
    "set": "O(1)",  # Setting a key-value pair
    "get": "O(1)",  # Retrieving a value by key
    "remove": "O(1)",  # Removing a key-value pair
    "clear": "O(n)",  # Clearing all key-value pairs
    "keys": "O(n)",  # Retrieving all keys
    "values": "O(n)",  # Retrieving all values
    "items": "O(n)",  # Retrieving all key-value pairs
    "pop": "O(1)",  # Removing a key and returning its value
    "popitem": "O(1)",  # Removing and returning the last inserted key-value pair
    "update": "O(len(other_dict))",  # Updating the dictionary with another dictionary
    "setdefault": "O(1)",  # Setting a default value for a key if it doesn't exist
}
stack_complexity = {"push": "O(1)", "pop": "O(1)", "peek": "O(1)"}
queue_complexity = {"enqueue": "O(1)", "dequeue": "O(1)", "peek": "O(1)"}

# Method selection and dynamic UI
if selected_ds == "List":
    ds = st.session_state.list
    methods = [
        "append",
        "insert",
        "remove",
        "get",
        "pop",
        "clear",
        "count",
        "reverse",
        "sort",
        "extend",
        "index",
        "reset",
    ]
    method = st.selectbox("Select Method", methods)
    code_snippets = list_code_snippets
    complexity = list_complexity

    if method == "append":
        value = st.text_input("Value to append:", key="list_append")
        if st.button("Execute"):
            ds.append(value)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "insert":
        col1, col2 = st.columns(2)
        with col1:
            index = st.number_input(
                "Index to insert at:", min_value=0, key="list_index"
            )
        with col2:
            insert_value = st.text_input("Value to insert:", key="list_insert")
        if st.button("Execute"):
            if ds.insert(int(index), insert_value):
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Invalid index!")

    elif method == "remove":
        remove_value = st.text_input("Value to remove:", key="list_remove")
        if st.button("Execute"):
            if ds.remove(remove_value):
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Value not found!")
    elif method == "pop":
        if st.button("Execute"):
            result = ds.pop()
            if result is not None:
                st.write(f"Popped value: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("List is empty!")

    elif method == "clear":
        if st.button("Execute"):
            ds.clear()
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "count":
        count_value = st.text_input("Value to count:", key="list_count")
        if st.button("Execute"):
            count = ds.count(count_value)
            st.write(f"Count of {count_value}: {count}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "reverse":
        if st.button("Execute"):
            ds.reverse()
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "sort":
        if st.button("Execute"):
            ds.sort()
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "extend":
        extend_value = st.text_input("Value to extend:", key="list_extend")
        if st.button("Execute"):
            ds.extend(extend_value)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")
    elif method == "index":
        index_value = st.text_input("Value to index:", key="list_index")
        if st.button("Execute"):
            index = ds.index(index_value)
            st.write(f"Index of {index_value}: {index}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "get":
        get_index = st.number_input("Index to get:", min_value=0, key="list_get")
        if st.button("Execute"):
            result = ds.get(int(get_index))
            if result is not None:
                st.write(f"Value at index {get_index}: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Index out of range!")

    elif method == "reset":
        if st.button("Execute"):
            st.session_state.list = MyList()
            st.success("List reset!")

    st.write("Current List State:", ds.get_state())

elif selected_ds == "Stack":
    ds = st.session_state.stack
    methods = ["push", "pop", "peek", "reset"]
    method = st.selectbox("Select Method", methods)
    code_snippets = stack_code_snippets
    complexity = stack_complexity

    if method == "push":
        value = st.text_input("Value to push:", key="stack_push")
        if st.button("Execute"):
            ds.push(value)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "pop":
        if st.button("Execute"):
            result = ds.pop()
            if result is not None:
                st.write(f"Popped: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Stack is empty!")

    elif method == "peek":
        if st.button("Execute"):
            result = ds.peek()
            if result is not None:
                st.write(f"Top: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Stack is empty!")

    elif method == "reset":
        if st.button("Execute"):
            st.session_state.stack = Stack()
            st.success("Stack reset!")

    st.write("Current Stack State (top first):", ds.get_state())

elif selected_ds == "Queue":
    ds = st.session_state.queue
    methods = ["enqueue", "dequeue", "peek", "reset"]
    method = st.selectbox("Select Method", methods)
    code_snippets = queue_code_snippets
    complexity = queue_complexity

    if method == "enqueue":
        value = st.text_input("Value to enqueue:", key="queue_enqueue")
        if st.button("Execute"):
            ds.enqueue(value)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "dequeue":
        if st.button("Execute"):
            result = ds.dequeue()
            if result is not None:
                st.write(f"Dequeued: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Queue is empty!")

    elif method == "peek":
        if st.button("Execute"):
            result = ds.peek()
            if result is not None:
                st.write(f"Front: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Queue is empty!")

    elif method == "reset":
        if st.button("Execute"):
            st.session_state.queue = Queue()
            st.success("Queue reset!")

    st.write("Current Queue State (left is front):", ds.get_state())

elif selected_ds == "Set":
    ds = st.session_state.set
    methods = [
        "add",  # Adds an element to the set
        "remove",  # Removes an element from the set
        "discard",  # Removes an element from the set without raising an error if it doesn't exist
        "contains",  # Checks if an element exists in the set
        "clear",  # Removes all elements from the set
        "pop",  # Removes and returns an arbitrary element from the set
        "copy",  # Creates a shallow copy of the set
        "union",  # Returns a new set with all elements from both sets
        "update",  # Adds elements from another set or iterable
        "intersection",  # Returns a new set with elements common to both sets
        "intersection_update",  # Updates the set with the intersection of itself and another
        "difference",  # Returns a new set with elements in the set but not in another
        "difference_update",  # Updates the set by removing elements found in another set
        "symmetric_difference",  # Returns a new set with elements in either set but not in both
        "symmetric_difference_update",  # Updates the set with the symmetric difference of itself and another
        "is_subset",  # Checks if the set is a subset of another
        "is_superset",  # Checks if the set is a superset of another
        "isdisjoint",  # Checks if two sets have no elements in common
        "reset",  # Resets the set to an empty state
    ]
    method = st.selectbox("Select Method", methods)
    code_snippets = set_code_snippets
    complexity = set_complexity

    if method == "add":
        value = st.text_input("Value to add:", key="set_add")
        if st.button("Execute"):
            ds.add(value)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "remove":
        remove_value = st.text_input("Value to remove:", key="set_remove")
        if st.button("Execute"):
            if ds.remove(remove_value):
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Value not in set!")

    elif method == "contains":
        check_value = st.text_input("Value to check:", key="set_contains")
        if st.button("Execute"):
            if ds.contains(check_value):
                st.write(f"{check_value} is in the set")
            else:
                st.write(f"{check_value} is not in the set")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "clear":
        if st.button("Execute"):
            ds.clear()
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "union":
        other_set = st.text_input("Other Set:", key="set_union")
        if st.button("Execute"):
            union_set = ds.union(other_set)
            st.write(f"Union Set: {union_set}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "intersection":
        other_set = st.text_input("Other Set:", key="set_intersection")
        if st.button("Execute"):
            intersection_set = ds.intersection(other_set)
            st.write(f"Intersection Set: {intersection_set}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")
    elif method == "difference":
        other_set = st.text_input("Other Set:", key="set_difference")
        if st.button("Execute"):
            difference_set = ds.difference(other_set)
            st.write(f"Difference Set: {difference_set}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")
    elif method == "symmetric_difference":
        other_set = st.text_input("Other Set:", key="set_symmetric_difference")
        if st.button("Execute"):
            symmetric_difference_set = ds.symmetric_difference(other_set)
            st.write(f"Symmetric Difference Set: {symmetric_difference_set}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")
    elif method == "is_subset":
        other_set = st.text_input("Other Set:", key="set_is_subset")
        if st.button("Execute"):
            is_subset = ds.is_subset(other_set)
            st.write(f"Is Subset: {is_subset}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")
    elif method == "is_superset":
        other_set = st.text_input("Other Set:", key="set_is_superset")
        if st.button("Execute"):
            is_superset = ds.is_superset(other_set)
            st.write(f"Is Superset: {is_superset}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")
    elif method == "is_disjoint":
        other_set = st.text_input("Other Set:", key="set_is_disjoint")
        if st.button("Execute"):
            is_disjoint = ds.is_disjoint(other_set)
            st.write(f"Is Disjoint: {is_disjoint}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "pop":
        if st.button("Execute"):
            popped_value = ds.pop()
            st.write(f"Popped Value: {popped_value}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "discard":
        discard_value = st.text_input("Value to discard:", key="set_discard")
        if st.button("Execute"):
            ds.discard(discard_value)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "copy":
        if st.button("Execute"):
            copied_set = ds.copy()
            st.write(f"Copied Set: {copied_set}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "update":
        update_set = st.text_input("Set to update:", key="set_update")
        if st.button("Execute"):
            ds.update(update_set)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "intersection_update":
        update_set = st.text_input("Set to update:", key="set_intersection_update")
        if st.button("Execute"):
            ds.intersection_update(update_set)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "difference_update":
        update_set = st.text_input("Set to update:", key="set_difference_update")
        if st.button("Execute"):
            ds.difference_update(update_set)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "symmetric_difference_update":
        update_set = st.text_input(
            "Set to update:", key="set_symmetric_difference_update"
        )
        if st.button("Execute"):
            ds.symmetric_difference_update(update_set)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "reset":
        if st.button("Execute"):
            st.session_state.set = MySet()
            st.success("Set reset!")

    st.write("Current Set State:", ds.get_state())

elif selected_ds == "Dictionary":
    ds = (
        st.session_state.dictionary
    )  # Ensure the dictionary object is initialized in session_state
    methods = [
        "set",  # Sets a key-value pair in the dictionary
        "get",  # Retrieves the value associated with a key
        "remove",  # Removes a key-value pair from the dictionary
        "clear",  # Removes all key-value pairs from the dictionary
        "keys",  # Returns a list of all keys in the dictionary
        "values",  # Returns a list of all values in the dictionary
        "items",  # Returns a list of all key-value pairs in the dictionary
        "pop",  # Removes a key and returns its value
        "popitem",  # Removes and returns the last inserted key-value pair
        "update",  # Updates the dictionary with another dictionary or iterable
        "setdefault",  # Returns the value of a key if it exists, otherwise sets it to a default value
    ]
    method = st.selectbox("Select Method", methods)
    code_snippets = (
        dict_code_snippets  # Dictionary containing code snippets for each method
    )
    complexity = (
        dict_complexity  # Dictionary containing time complexities for each method
    )

    if method == "set":
        col1, col2 = st.columns(2)
        with col1:
            key = st.text_input("Key to set:", key="dict_key")
        with col2:
            value = st.text_input("Value to set:", key="dict_value")
        if st.button("Execute"):
            ds.set(key, value)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "get":
        get_key = st.text_input("Key to get:", key="dict_get")
        if st.button("Execute"):
            result = ds.get(get_key)
            if result is not None:
                st.write(f"Value for {get_key}: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Key not found!")

    elif method == "remove":
        remove_key = st.text_input("Key to remove:", key="dict_remove")
        if st.button("Execute"):
            if ds.remove(remove_key):
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Key not found!")

    elif method == "clear":
        if st.button("Execute"):
            ds.clear()
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "keys":
        if st.button("Execute"):
            keys = ds.keys()
            st.write(f"Keys: {keys}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "values":
        if st.button("Execute"):
            values = ds.values()
            st.write(f"Values: {values}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")
    elif method == "items":
        if st.button("Execute"):
            items = ds.items()
            st.write(f"Items: {items}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "copy":
        if st.button("Execute"):
            copied_dict = ds.copy()
            st.write(f"Copied Dictionary: {copied_dict}")
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "update":
        update_dict = st.text_input("Dictionary to update:", key="dict_update")
        if st.button("Execute"):
            ds.update(update_dict)
            st.code(code_snippets[method], language="python")
            st.write(f"Time Complexity: {complexity[method]}")

    elif method == "pop":
        pop_key = st.text_input("Key to pop:", key="dict_pop")
        if st.button("Execute"):
            result = ds.pop(pop_key)
            if result is not None:
                st.write(f"Popped value: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Key not found!")

    elif method == "popitem":
        if st.button("Execute"):
            result = ds.popitem()
            if result is not None:
                st.write(f"Popped item: {result}")
                st.code(code_snippets[method], language="python")
                st.write(f"Time Complexity: {complexity[method]}")
            else:
                st.error("Dictionary is empty!")

    elif method == "reset":
        if st.button("Execute"):
            st.session_state.dict = MyDict()
            st.success("Dictionary reset!")

    st.write("Current Dictionary State:", ds.get_state())


# Sidebar display of current state
st.sidebar.subheader("Current State")
st.sidebar.write(
    f"{selected_ds} State:", getattr(st.session_state, selected_ds.lower()).get_state()
)
