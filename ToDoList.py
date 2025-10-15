def load_tasks(filename="tasks.txt"):
    """
    Reads tasks from a file.
    Returns a list of tasks, or an empty list if the file doesn't exist.
    """
    try:
        with open(filename, 'r') as file:
            # Read each line and strip the newline character from the end
            tasks = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        tasks = []
    return tasks

def save_tasks(tasks, filename="tasks.txt"):
    """
    Saves a list of tasks to a file, overwriting the existing file.
    """
    with open(filename, 'w') as file:
        for task in tasks:
            file.write(task + '\n')
            
def view_tasks(tasks):
    """Prints all tasks in a numbered list."""
    if not tasks:
        print("Your to-do list is empty.")
    else:
        print("\n--- To-Do List ---")
        for i, task in enumerate(tasks, start=1):
            print(f"{i}. {task}")
        print("------------------")

def add_task(tasks):
    """Asks the user for a new task and adds it to the list."""
    task = input("Enter the new task: ")
    tasks.append(task)
    
    print(f"Task '{task}' added.")
    return tasks

def remove_task(tasks):
    """Asks user for a task number and removes it from the list."""
    if not tasks:
        print("Your to-do list is empty. Nothing to remove.")
        return tasks

    view_tasks(tasks)
    
    try:
        # Ask the user for the number of the task to remove
        task_num_str = input("Enter the number of the task you want to remove: ")
        # Convert the input string to an integer
        task_num = int(task_num_str)

        # Convert from 1-based (for the user) to 0-based (for the list index)
        index = task_num - 1

        # Check if the index is valid
        if 0 <= index < len(tasks):
            removed_task = tasks.pop(index)
            print(f"Task '{removed_task}' removed.")
        else:
            print("Invalid task number.")
            
    except ValueError:
        # This block runs if the user enters something that isn't a number
        print("Invalid input. Please enter a number.")

    return tasks
            
    
    
def main():
    tasks = load_tasks()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. View Tasks")
        print("2. Add a task")
        print("3. Remove a task")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            view_tasks(tasks)
        elif choice == '2':
            tasks = add_task(tasks)
        elif choice == '3':
            tasks = remove_task(tasks)
        elif choice == '4':
            save_tasks(tasks)
            print("To-do-list saved. Goodbye!")
            break 
        else:
            print("Invalid")
            
main()
    