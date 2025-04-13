# Encapsulation Excersice

# Create a BankAccount class with deposit() and withdraw() methods.


class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.__balance = balance

    def deposit(self, amount):

        if amount > 0:
            self.__balance += amount
            print(f"Deposited {amount}. New balance: {self.__balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew {amount}. New balance: {self.__balance}")
        else:
            print("Insufficient funds or invalid amount.")

    def get_balance(self):
        return self.__balance


account = BankAccount("12345678", 1000)

account.deposit(500)
account.withdraw(200)
print(f"Current balance: {account.get_balance()}")


# Inheritance Excersice

# Implement an Employee class that inherits from a Person class.


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        print(f"{self.name} is speaking.")


class Employee(Person):
    def __init__(self, name, age, job_title):
        super().__init__(name, age)
        self.job_title = job_title

    def work(self):
        print(f"{self.name} is working as a {self.job_title}.")


employee = Employee("Alice", 30, "Software Engineer")
employee.speak()  # Output: Alice is speaking.
employee.work()  # Output: Alice is working as a Software Engineer


# Inheritance Excersice

# Override the speak() method in different child classes (Polymorphism).


class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("Animal is making a sound.")


class Dog(Animal):
    def speak(self):
        print("Dog is barking.")


class Cat(Animal):
    def speak(self):
        print("Cat is meowing.")


animals = [Dog("Fido"), Cat("Whiskers")]

for animal in animals:
    animal.speak()


# Abstraction Excersice

# 📌 Steps to Follow:
# 1️⃣ Create an abstract class BankAccount with:

# An __init__ method to store account_holder and balance.

# An abstract method calculate_interest() (to be implemented by child classes).

# 2️⃣ Create two child classes:

# SavingsAccount (interest rate: 5%).

# CurrentAccount (no interest).

# 3️⃣ Instantiate objects of both accounts and print the interest.

from abc import ABC, abstractmethod


class BankAccount(ABC):
    def __init__(self, account_holder, balance):
        self.account_holder = account_holder
        self.balance = balance

    @abstractmethod
    def calculate_interest(self):
        pass

    def display_balance(self):
        print(f"Account Holder: {self.account_holder}")
        print(f"Balance: {self.balance}")
        print(f"Account Type: {self.__class__.__name__}")
        print(f"Interest: {self.calculate_interest()}")


class SavingsAccount(BankAccount):
    def calculate_interest(self):
        return self.balance * 0.05


class CurrentAccount(BankAccount):
    def calculate_interest(self):
        return 0


savings_account = SavingsAccount("John Doe", 1000)
current_account = CurrentAccount("Jane Doe", 500)

savings_account.calculate_interest()
current_account.calculate_interest()

savings_account.display_balance()
current_account.display_balance()
