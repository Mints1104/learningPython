class Vehicle:
    def __init__(self, make, model):
        self.make = make
        self.model = model
    
    def get_info(self):
        return f"{self.make}{self.model}"
    