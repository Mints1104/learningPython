from Node import Node
class Stack:
    def __init__(self):
        self.top = None
    
    def push(self,value):
       new_node = Node(value)
       new_node.next = self.top
       self.top = new_node
    def pop(self):
        if self.top is None:
            return None
        popped_value = self.top.value
        self.top = self.top.next
        return  popped_value


my_stack = Stack()
my_stack.push(10)
my_stack.push(20)
print(my_stack.pop())
print(my_stack.pop())
print(my_stack.pop())