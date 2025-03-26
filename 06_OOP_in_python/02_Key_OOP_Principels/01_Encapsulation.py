# 📌 Encapsulation
"""
Encapsulation means bundling data (attributes) and methods into a single unit (class) and controlling access to that data. 
It protects the internal state of an object by making some attributes private.

### Key Points:

1. **Public Attributes**: Accessible from anywhere (e.g., `self.name`).
2. **Private Attributes**: Prefixed with `__` (e.g., `self.__balance`), only accessible within the class.

### Example: BankAccount Class

The `BankAccount` class demonstrates encapsulation by:
- Using private attributes to protect sensitive data (`__balance`).
- Providing public methods (`deposit`, `withdraw`, `get_balance`) to control access to private data.
"""

class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number  # Public attribute
        self.__balance = balance              # Private attribute

    def deposit(self, amount):
        """Deposit money into the account."""
        if amount > 0:
            self.__balance += amount
            print(f"Deposited {amount}. New balance: {self.__balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        """Withdraw money from the account."""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew {amount}. New balance: {self.__balance}")
        else:
            print("Insufficient funds or invalid amount.")

    def get_balance(self):
        """Public method to access the private balance."""
        return self.__balance

# Usage Example
if __name__ == "__main__":
    # Create a BankAccount object
    account = BankAccount("12345678", 1000)

    # Perform operations
    account.deposit(500)      # Output: Deposited 500. New balance: 1500
    account.withdraw(200)     # Output: Withdrew 200. New balance: 1300
    print(f"Current balance: {account.get_balance()}")  # Output: Current balance: 1300

    # Trying to access private attribute directly (will fail)
    # print(account.__balance)  # AttributeError: 'BankAccount' object has no attribute '__balance'

"""
### Explanation:
1. **Private Attribute (`__balance`)**:
   - The `__balance` attribute is private and cannot be accessed directly outside the class.
   - This ensures the balance cannot be modified accidentally or maliciously.

2. **Public Methods**:
   - `deposit`: Allows controlled addition of money to the account.
   - `withdraw`: Allows controlled withdrawal of money, ensuring sufficient funds.
   - `get_balance`: Provides read-only access to the private balance.

### Benefits of Encapsulation:
- Protects sensitive data from unauthorized access or modification.
- Ensures controlled access to attributes through public methods.
- Makes the code modular, reusable, and easier to maintain.
"""