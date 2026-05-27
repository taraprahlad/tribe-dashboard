"""to-do list manager:
i'm making a pretty basic to-do list manager just to brush up on oop stuff
we learned in 15-112 lmao -- i know it's a little unforgiving when it comes to spaces
but i'm too lazy to fix it and this is just practice anyway so who gives a shit"""

class Task:
    # need a description of the task and whether or not it's been completed
    def __init__(self, description):
        self.description = description
        self.done = False

    def mark_done(self): #switch to complete
        self.done = True

    def __str__(self): #what it looks like
        checkbox = "x" if self.done else " "
        return f"[{checkbox} {self.description}]"
    
#our to-do list is a collection of task objects
class ToDoList:
    def __init__(self):
        self.tasks = [] #starts empty and will be filled with task objects as we go along

    #the to-do list needs to add tasks, delete them, mark them complete and number them (for extra practice lol)

    def add(self, description):
        self.tasks.append(Task(description))
    
    def delete(self, number):
        if 1 <= number <= len(self.tasks):
            self.tasks.pop(number - 1)
        else:
            print("gurl that task number doesn't exist tf")
    
    def complete(self, number):
        if 1 <= number <= len(self.tasks):
            self.tasks[number - 1].mark_done()
        else:
            print("gurl that task number doesn't exist tf")
    
    def numberdisplay(self):
        if not self.tasks:
            print("there's nothing to do")
        else:
            for i, task in enumerate(self.tasks, start=0):
                print(f"{i}. {task}")
            
def main(): #let's run ts
    todo = ToDoList()
    while True:
        print("\n--- TO-DO List ---")
        todo.numberdisplay() 
        print("\nOptions: click 'a' to add a task, 'b' to remove, 'c' to mark as done and 'd' to quit")
        choice = input("what do you want to do? >>> ")

        if choice == "a":
            description = input("task description: ")
            if description:
                todo.add(description)
        elif choice == "b":
            number = int(input("which task number do you want to delete?"))
            todo.delete(number)
        elif choice == "c":
            number = int(input("what task number did you finish?"))
            todo.complete(number)
            print("yayyyy nice work!")
        elif choice == "d":
            print("see you later :) ")
            break
        else:
            choice = input("type something coherent please >>> ")

main()





    



