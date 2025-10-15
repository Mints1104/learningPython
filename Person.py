from Car import Car 

class Person:
    def __init__(self,name,car_make,car_model,car_year):
        self.name = name
        self.car = Car(car_make,car_model,car_year)
    
    def introduce(self):
        print(f"Hi, my name is {self.name}. I drive a:")
        self.car.display_info()
    
person1 = Person("Harun", "Ford","Focus",2021)
person1.introduce()