class Task:
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.status = "pending"

    def mark_complete(self):
        self.status = "completed"
        print(f"Task '{self.title}' marked as completed.")


class ToDoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)
        print(f"Task '{task.title}' added.")

    def remove_task(self, title):
        for task in self.tasks:
            if task.title == title:
                self.tasks.remove(task)
                print(f"Task '{title}' removed.")
                return
        print(f"Task '{title}' not found.")

    def display_tasks(self):
        if not self.tasks:
            print("No tasks in the list.")
        else:
            for task in self.tasks:
                print(
                    f"Title: {task.title}, Description: {task.description}, Status: {task.status}"
                )


# Usage
todo = ToDoList()
task1 = Task("Buy groceries", "Milk, eggs, bread")
task2 = Task("Read a book", "Finish novel")

todo.add_task(task1)
todo.add_task(task2)
todo.display_tasks()

todo.remove_task("Buy groceries")

todo.display_tasks()
