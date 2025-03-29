import streamlit as st
from collections import deque

# Data structure classes (from Step 2)
class MyList:
    def __init__(self):
        self.items = []
    def append(self, value):
        self.items.append(value)
    def insert(self, index, value):
        if 0 <= index <= len(self.items):
            self.items.insert(index, value)
            return True
        return False
    def remove(self, value):
        if value in self.items:
            self.items.remove(value)
            return True
        return False
    def get(self, index):
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
    def get_state(self):
        return self.items

class Stack:
    def __init__(self):
        self.items = []
    def push(self, value):
        self.items.append(value)
    def pop(self):
        if self.items:
            return self.items.pop()
        return None
    def peek(self):
        if self.items:
            return self.items[-1]
        return None
    def get_state(self):
        return self.items[::-1]

class Queue:
    def __init__(self):
        self.items = deque()
    def enqueue(self, value):
        self.items.append(value)
    def dequeue(self):
        if self.items:
            return self.items.popleft()
        return None
    def peek(self):
        if self.items:
            return self.items[0]
        return None
    def get_state(self):
        return list(self.items)

class MySet:
    def __init__(self):
        self.items = set()
    def add(self, value):
        self.items.add(value)
    def remove(self, value):
        if value in self.items:
            self.items.remove(value)
            return True
        return False
    def contains(self, value):
        return value in self.items
    def get_state(self):
        return list(self.items)

class MyDict:
    def __init__(self):
        self.items = {}
    def set(self, key, value):
        self.items[key] = value
    def get(self, key):
        return self.items.get(key, None)
    def remove(self, key):
        if key in self.items:
            del self.items[key]
            return True
        return False
    def get_state(self):
        return self.items

# Streamlit app
st.title("🧪 Data Structure Playground")
st.write("Explore and experiment with core data structures in Python.")

# Initialize session state
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

# Select data structure
data_structures = ["List", "Stack", "Queue", "Set", "Dictionary"]
selected_ds = st.selectbox("Choose a Data Structure", data_structures)

# UI for each data structure
if selected_ds == "List":
    ds = st.session_state.list
    st.subheader("List Operations")
    st.write("A dynamic array for storing ordered elements.")
    
    value = st.text_input("Value to append:", key="list_append")
    if st.button("Append", key="list_append_btn"):
        ds.append(value)
        st.success(f"Appended {value}")
    
    col1, col2 = st.columns(2)
    with col1:
        index = st.number_input("Index to insert:", min_value=0, step=1, key="list_insert_idx")
    with col2:
        insert_value = st.text_input("Value to insert:", key="list_insert_val")
    if st.button("Insert", key="list_insert_btn"):
        if ds.insert(int(index), insert_value):
            st.success(f"Inserted {insert_value} at {index}")
        else:
            st.error("Invalid index")
    
    remove_value = st.text_input("Value to remove:", key="list_remove")
    if st.button("Remove", key="list_remove_btn"):
        if ds.remove(remove_value):
            st.success(f"Removed {remove_value}")
        else:
            st.error("Value not found")
    
    get_index = st.number_input("Index to get:", min_value=0, step=1, key="list_get_idx")
    if st.button("Get", key="list_get_btn"):
        result = ds.get(int(get_index))
        if result is not None:
            st.write(f"Value at {get_index}: {result}")
        else:
            st.error("Invalid index")
    
    if st.button("Reset List", key="list_reset"):
        st.session_state.list = MyList()
        st.success("List reset")
    
    st.write("**Current List:**", ds.get_state())

elif selected_ds == "Stack":
    ds = st.session_state.stack
    st.subheader("Stack Operations")
    st.write("A LIFO (Last In, First Out) structure.")
    
    value = st.text_input("Value to push:", key="stack_push")
    if st.button("Push", key="stack_push_btn"):
        ds.push(value)
        st.success(f"Pushed {value}")
    
    if st.button("Pop", key="stack_pop_btn"):
        result = ds.pop()
        if result is not None:
            st.success(f"Popped {result}")
        else:
            st.error("Stack is empty")
    
    if st.button("Peek", key="stack_peek_btn"):
        result = ds.peek()
        if result is not None:
            st.write(f"Top: {result}")
        else:
            st.error("Stack is empty")
    
    if st.button("Reset Stack", key="stack_reset"):
        st.session_state.stack = Stack()
        st.success("Stack reset")
    
    st.write("**Current Stack (top to bottom):**", ds.get_state())

elif selected_ds == "Queue":
    ds = st.session_state.queue
    st.subheader("Queue Operations")
    st.write("A FIFO (First In, First Out) structure.")
    
    value = st.text_input("Value to enqueue:", key="queue_enqueue")
    if st.button("Enqueue", key="queue_enqueue_btn"):
        ds.enqueue(value)
        st.success(f"Enqueued {value}")
    
    if st.button("Dequeue", key="queue_dequeue_btn"):
        result = ds.dequeue()
        if result is not None:
            st.success(f"Dequeued {result}")
        else:
            st.error("Queue is empty")
    
    if st.button("Peek", key="queue_peek_btn"):
        result = ds.peek()
        if result is not None:
            st.write(f"Front: {result}")
        else:
            st.error("Queue is empty")
    
    if st.button("Reset Queue", key="queue_reset"):
        st.session_state.queue = Queue()
        st.success("Queue reset")
    
    st.write("**Current Queue (front to rear):**", ds.get_state())

elif selected_ds == "Set":
    ds = st.session_state.set
    st.subheader("Set Operations")
    st.write("An unordered collection of unique elements.")
    
    value = st.text_input("Value to add:", key="set_add")
    if st.button("Add", key="set_add_btn"):
        ds.add(value)
        st.success(f"Added {value}")
    
    remove_value = st.text_input("Value to remove:", key="set_remove")
    if st.button("Remove", key="set_remove_btn"):
        if ds.remove(remove_value):
            st.success(f"Removed {remove_value}")
        else:
            st.error("Value not found")
    
    check_value = st.text_input("Value to check:", key="set_check")
    if st.button("Contains", key="set_contains_btn"):
        result = ds.contains(check_value)
        st.write(f"Contains {check_value}: {result}")
    
    if st.button("Reset Set", key="set_reset"):
        st.session_state.set = MySet()
        st.success("Set reset")
    
    st.write("**Current Set:**", ds.get_state())

elif selected_ds == "Dictionary":
    ds = st.session_state.dict
    st.subheader("Dictionary Operations")
    st.write("A collection of key-value pairs.")
    
    col1, col2 = st.columns(2)
    with col1:
        key = st.text_input("Key:", key="dict_key")
    with col2:
        value = st.text_input("Value:", key="dict_value")
    if st.button("Set", key="dict_set_btn"):
        ds.set(key, value)
        st.success(f"Set {key}: {value}")
    
    get_key = st.text_input("Key to get:", key="dict_get")
    if st.button("Get", key="dict_get_btn"):
        result = ds.get(get_key)
        if result is not None:
            st.write(f"Value for {get_key}: {result}")
        else:
            st.error("Key not found")
    
    remove_key = st.text_input("Key to remove:", key="dict_remove")
    if st.button("Remove", key="dict_remove_btn"):
        if ds.remove(remove_key):
            st.success(f"Removed {remove_key}")
        else:
            st.error("Key not found")
    
    if st.button("Reset Dictionary", key="dict_reset"):
        st.session_state.dict = MyDict()
        st.success("Dictionary reset")
    
    st.write("**Current Dictionary:**", ds.get_state())