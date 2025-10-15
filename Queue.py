from Node import Node
class Queue:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, value):
        new_node = Node(value)
        if self.tail is None:
            self.head = new_node
            self.tail = new_node
            return
        self.tail.next = new_node
        self.tail = new_node
        
        

    def dequeue(self):
        
        if self.head is None:
            return None
        dequeued_value = self.head.value
        self.head = self.head.next
        if self.head is None:
            self.tail = None
        return dequeued_value
       
        
        