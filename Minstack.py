class MinStack:
    def __init__(self):
        # The main stack to store all values
        self.stack = []
        # The min_stack to store the minimums. The last element is always the current min.
        self.min_stack = []

    def push(self, val):
        # Always push the new value onto the main stack
        self.stack.append(val)
        
        # Only push onto the min_stack if it's a new minimum
        # (or if the min_stack is empty)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        # Check if the value we are about to pop from the main stack
        # is also the current minimum.
        if self.stack[-1] == self.min_stack[-1]:
            # If so, we must pop it from the min_stack as well.
            self.min_stack.pop()
        
        # Always pop from the main stack
        self.stack.pop()

    def top(self):
        # The top is just the last element in the main stack
        return self.stack[-1]

    def getMin(self):
        # The current minimum is always the last element in the min_stack
        return self.min_stack[-1]

# --- Example Usage ---
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin()) # return -3
minStack.pop()
print(minStack.top())    # return 0
print(minStack.getMin()) # return -2