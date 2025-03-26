from abc import ABC, abstractmethod

# Abstract Bank Account Class
class Bank_Account(ABC):
    def __init__(self, account_holder, account_number, balance=0.0):
        self._account_holder = account_holder
        self.account_number = account_number
        self._balance = balance  # 🔥 Private attribute
    
    @abstractmethod
    def calculate_interest(self):
        pass  # Abstract method to be implemented in subclasses

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            print(f"✅ Deposited {amount}. New balance: {self._balance}")
        else:
            print("❌ Invalid deposit amount.")
    
    def withdraw(self, amount):
        if amount > 0 and amount <= self._balance:
            self._balance -= amount
            print(f"✅ Withdrew {amount}. New balance: {self._balance}")
        else:
            print("❌ Insufficient funds or invalid amount.")
    
    def get_balance(self):  # ✅ Getter method to access private balance
        return self._balance
    
    def display_balance(self):
        print("\n🔹 Account Details 🔹")
        print(f"🏦 Account Holder: {self._account_holder}")
        print(f"🔢 Account Number: {self.account_number}")
        print(f"💰 Balance: ${self._balance}")
        print(f"📌 Account Type: {self.__class__.__name__}")
        print(f"💵 Interest Earned: ${self.calculate_interest()}")
        print("-" * 30)

# Savings Account (Inherits from Bank_Account)
class SavingsAccount(Bank_Account):
    def calculate_interest(self):
        return self._balance * 0.05  # 5% Interest Rate

# Current Account (Inherits from Bank_Account)
class CurrentAccount(Bank_Account):
    def calculate_interest(self):
        return self._balance * 0.01  # 1% Interest Rate

# Employee Class.
class Employee:
    def __init__(self, name, age, job_title, salary, bank_account):
        self.name = name
        self.age = age
        self.job_title = job_title
        self.salary = salary
        self.bank_account = bank_account
    
    def display_info(self):
        print("\n🔹 Employee Details 🔹")
        print(f"👤 Name: {self.name}")
        print(f"🎂 Age: {self.age}")
        print(f"💼 Job Title: {self.job_title}")
        print(f"💰 Salary: ${self.salary}")
        print(f"🏦 Bank Balance: ${self.bank_account.get_balance()}")
        print(f"💵 Interest Earned: ${self.bank_account.calculate_interest()}")
        print("-" * 30)

    def deposit_to_account(self, amount):
        self.bank_account.deposit(amount)
        
    def withdraw_from_account(self, amount):
        self.bank_account.withdraw(amount)

# Manager Class (Inherits from Employee)
class Manager(Employee):
    def __init__(self, name, age, job_title, salary, department, bank_account):
        super().__init__(name, age, job_title, salary, bank_account)
        self.department = department

    def display_info(self):
        super().display_info()
        print(f"📂 Department: {self.department}")
        print("-" * 30)

# 🔥 Demonstration
# ✅ Creating Bank Accounts
john_account = SavingsAccount("John Doe", "12345", 0)
jane_account = SavingsAccount("Jane Smith", "67890", 0)
alice_account = CurrentAccount("Alice Johnson", "54321", 3000)

# ✅ Creating Employees & Manager
john = Employee("John Doe", 30, "Software Engineer", 5000, john_account)
jane = Manager("Jane Smith", 40, "Manager", 8000, "IT", jane_account)
alice = Employee("Alice Johnson", 28, "Data Analyst", 6000, alice_account)

# ✅ Performing Transactions
john.deposit_to_account(2000)
jane.deposit_to_account(5000)
alice.withdraw_from_account(500)

# ✅ Displaying Employee Details
print("\n🎯 Employee Information:")
john.display_info()
jane.display_info()
alice.display_info()

# ✅ Displaying Account Details
print("\n📌 Account Summaries:")
john_account.display_balance()
jane_account.display_balance()
alice_account.display_balance()
