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
if 'list' not in st.session_state:
    st.session_state.list = MyList()
if 'stack' not in st.session_state:
    st.session_state.stack = Stack()
if 'queue' not in st.session_state:
    st.session_state.queue = Queue()
if 'set' not in st.session_state:
    st.session_state.set = MySet()
if 'dict' not in st.session_state:
    st.session_state.dict = MyDict()

# Data structure selection in sidebar
data_structures = ["List", "Stack", "Queue", "Set", "Dictionary"]
selected_ds = st.sidebar.selectbox("Choose Data Structure", data_structures)

# Code snippets and complexities
list_code_snippets = {
    "append": "def append(self, value):\n    self.items.append(value)\n    return True",
    "insert": "def insert(self, index, value):\n    if 0 <= index <= len(self.items):\n        self.items.insert(index, value)\n        return True\n    return False",
    "remove": "def remove(self, value):\n    if value in self.items:\n        self.items.remove(value)\n        return True\n    return False",
    "get": "def get(self, index):\n    if 0 <= index < len(self.items):\n        return self.items[index]\n    return None"
}
stack_code_snippets = {
    "push": "def push(self, value):\n    self.items.append(value)",
    "pop": "def pop(self):\n    if not self.is_empty():\n        return self.items.pop()\n    return None",
    "peek": "def peek(self):\n    if not self.is_empty():\n        return self.items[-1]\n    return None"
}
queue_code_snippets = {
    "enqueue": "def enqueue(self, value):\n    self.items.append(value)",
    "dequeue": "def dequeue(self):\n    if not self.is_empty():\n        return self.items.popleft()\n    return None",
    "peek": "def peek(self):\n    if not self.is_empty():\n        return self.items[0]\n    return None"
}
set_code_snippets = {
    "add": "def add(self, value):\n    self.items.add(value)",
    "remove": "def remove(self, value):\n    if value in self.items:\n        self.items.remove(value)\n        return True\n    return False",
    "contains": "def contains(self, value):\n    return value in self.items"
}
dict_code_snippets = {
    "set": "def set(self, key, value):\n    self.items[key] = value",
    "get": "def get(self, key):\n    return self.items.get(key)",
    "remove": "def remove(self, key):\n    if key in self.items:\n        del self.items[key]\n        return True\n    return False"
}

list_complexity = {"append": "O(1)", "insert": "O(n)", "remove": "O(n)", "get": "O(1)"}
stack_complexity = {"push": "O(1)", "pop": "O(1)", "peek": "O(1)"}
queue_complexity = {"enqueue": "O(1)", "dequeue": "O(1)", "peek": "O(1)"}
set_complexity = {"add": "O(1)", "remove": "O(1)", "contains": "O(1)"}
dict_complexity = {"set": "O(1)", "get": "O(1)", "remove": "O(1)"}

# Method selection and dynamic UI
if selected_ds == "List":
    ds = st.session_state.list
    methods = ["append", "insert", "remove", "get", "reset"]
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
            index = st.number_input("Index to insert at:", min_value=0, key="list_index")
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
    methods = ["add", "remove", "contains", "reset"]
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
    
    elif method == "reset":
        if st.button("Execute"):
            st.session_state.set = MySet()
            st.success("Set reset!")

    st.write("Current Set State:", ds.get_state())

elif selected_ds == "Dictionary":
    ds = st.session_state.dict
    methods = ["set", "get", "remove", "reset"]
    method = st.selectbox("Select Method", methods)
    code_snippets = dict_code_snippets
    complexity = dict_complexity

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
    
    elif method == "reset":
        if st.button("Execute"):
            st.session_state.dict = MyDict()
            st.success("Dictionary reset!")

    st.write("Current Dictionary State:", ds.get_state())

# Sidebar display of current state
st.sidebar.subheader("Current State")
st.sidebar.write(f"{selected_ds} State:", getattr(st.session_state, selected_ds.lower()).get_state())