from Player import Player
class Room:
    def __init__(self,name,description,item):
        self.name = name
        self.description = description 
        self.item = item
        self.exits = {}
        
    def link_room(self,room,direction):
        self.exits[direction] = room
        
   
    def collect_item(self,player,item):
        item = self.item 
        self.item = None
        player.add_item(item)

    def travel_between_room(self,player,direction):
       new_room =  self.exits[direction]
       if new_room.item is None:
           print("Item has already been collected")
       else:
           print(f"Collecting item: {new_room.item}")
           self.collect_item(player,new_room.item)
       return new_room
            



def main():
    player = Player()
    kitchen = Room("Kitchen","You are in a kitchen. It's a bit messy.","Pan")
    dining_room = Room("Dining Room","You are in a grand dining room. A large table is in the center.","Plate")
    hallway = Room("Hallway","You are in a long hallway.","Key")

    # Link the rooms together
    kitchen.link_room(dining_room, "north")
    dining_room.link_room(kitchen, "south")
    dining_room.link_room(hallway, "east")
    hallway.link_room(dining_room, "west")
    valid_directions = ["north","east","south","west"]
    currentRoom = kitchen
    while True:
        print(f"Current Room: {currentRoom.name}")
        print("What direction would you like to travel in?")
        direction = input()
        if direction not in valid_directions:
            print("Invalid direction, please choose a valid direction.")
        else:
            print(f"Chosen direction: {direction}")
            currentRoom =  currentRoom.travel_between_room(player,direction)
            
        
        
main()