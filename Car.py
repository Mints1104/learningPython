from Vehicle import Vehicle
class Car(Vehicle):
    def __init__(self, make, model, num_doors):
        super().__init__(make, model)
        self.num_doors = num_doors
    
my_car = Car("Ford","Focus",4)
print(my_car.get_info())
print(f"Doors: {my_car.num_doors}")