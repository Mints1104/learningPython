class Player:
    def __init__(self):
        self.inventory = []
        
    def add_item(self,item):
        self.inventory.append(item)
        