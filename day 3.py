#task 1 class creation
class Dog:
    def __init__(self, name, age):
        self.name, self.age = name, age

    def bark(self): return f"{self.name} is barking"
    def sleep(self): return f"{self.name} is sleeping..."


dog1 = Dog("Buddy", 3)
print(dog1.bark())   
print(dog1.sleep())  


#task 2 constructor
class Dog:
    def __init__(self, name, age):
        self.name,self.age = name,age
dog1 = Dog("Buddy", 3)
print(f"Dog's Name: {dog1.name}, Age: {dog1.age}")


#task 3 inheritence
class Dog:
    def __init__(self, name, age):
        self.name, self.age = name, age

    def bark(self):
        return f"{self.name} is barking"
    def sleep(self):
        return f"{self.name} is sleeping"

class Puppy(Dog):
    def __init__(self, name, age, breed):
        super().__init__(name, age)  
        self.breed = breed 

    def play(self):
        return f"{self.name} is playing happily!"


puppy1 = Puppy("Charlie", 1, "Labrador")
print(f"Puppy's Name: {puppy1.name}, Age: {puppy1.age}, Breed: {puppy1.breed}")  
print(puppy1.bark())
print(puppy1.sleep())
print(puppy1.play())  
