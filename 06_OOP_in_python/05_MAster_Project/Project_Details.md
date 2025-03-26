# 💼 Unified Assignment: Bank & Employee Management System

## 📌 Scenario:
You're building a **Bank & Employee Management System** where:
- Employees have bank accounts (every employee has an account).
- There are different types of accounts (Savings, Current).
- The system should handle transactions, calculate interest, and manage employees.

---

## 📝 Step-by-Step Implementation:

### Step 1: Abstract `BankAccount` Class (Encapsulation + Abstraction)
- **Use the `ABC` module** to create an abstract class.
- **Attributes** (Encapsulation - private attributes):
  - `_account_holder` (str)
  - `_balance` (float)
- **Abstract Method**:
  - `calculate_interest()`
- **Methods**:
  - `deposit(amount)`: Increases balance.
  - `withdraw(amount)`: Decreases balance.
  - `get_balance()`: Returns balance.

---

### Step 2: `SavingsAccount` & `CurrentAccount` (Inheritance + Polymorphism)
- **SavingsAccount**:
  - Interest rate: 5%
- **CurrentAccount**:
  - No interest
- **Override `calculate_interest()`** in both classes.

---

### Step 3: `Employee` Class (Inheritance + Encapsulation)
- **Attributes**:
  - `name`, `age`, `job_title`, `_salary` (private).
- **Method**:
  - `display_info()`: Prints employee details.

---

### Step 4: `Manager` Class (Inheritance + Polymorphism)
- Inherits from `Employee`.
- **Additional Attribute**:
  - `department`
- **Override `display_info()`** to include department details.

---

### Step 5: Employee-Account Relationship (Composition)
- Every Employee must have a **Bank Account**.
- Store an instance of a `BankAccount` inside the `Employee` class.
- Employees can **deposit** and **withdraw** from their own accounts.

---

### Final Step: Demonstration (Polymorphism in Action)
- Create multiple **Employees** and **Managers**.
- Assign them different bank accounts.
- Perform transactions & show balances.
- Call `display_info()` on all employees.

---

## 💡 Expected Example Output:

### Employee Details:


John Doe (Software Engineer) - Salary: $5000 Bank Balance: $2000 Interest Earned: $100.0

Jane Smith (Manager) - Department: IT - Salary: $8000 Bank Balance: $5000 Interest Earned: $250.0