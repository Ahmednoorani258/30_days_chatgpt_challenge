# 🚀 Day 23: Enhance Your Code Quality

## 🎯 Goal
Implement best practices for writing clean, maintainable Python code by adding automated testing, linters, and continuous integration (CI). By the end of today, you’ll have a project repository that enforces quality checks on every change.

---

## 📚 Why Code Quality Matters
1. **Reliability**: Automated tests catch regressions early.
2. **Consistency**: Linters enforce style guides so your codebase looks uniform.
3. **Collaboration**: CI ensures that every PR meets quality standards before merging.

---

## 🧩 Step-by-Step Tasks

### 1️⃣ Set Up a Testing Framework
- **Choose a Framework**: `pytest` is the de‑facto standard in Python.

#### Install:
# pip install pytest

#### Write Your First Test:
1. Create a `tests/` directory.
2. Add a file `tests/test_example.py`:
   # from my_module import add
   #
   # def test_add_positive_numbers():
   #     assert add(2, 3) == 5
   #
   # def test_add_negative_numbers():
   #     assert add(-1, -1) == -2

#### Run Tests:
# pytest --maxfail=1 --disable-warnings -q

---

### 2️⃣ Add Code Coverage Measurement
#### Install Coverage:
# pip install coverage

#### Run with Coverage:
# coverage run -m pytest
# coverage report -m

- **Goal**: Aim for ≥ 80% coverage on your core modules.

---

### 3️⃣ Integrate a Linter and Formatter
- **Choose Tools**:
  - `flake8` for linting.
  - `black` for automatic code formatting.

#### Install:
# pip install flake8 black

#### Configure `flake8`:
Create a `.flake8` file:
# [flake8]
# max-line-length = 88
# extend-ignore = E203, W503

#### Run Linters & Formatter:
# black .        # formats code
# flake8         # reports style errors

---

### 4️⃣ Add Type Checking with `mypy`
#### Install:
# pip install mypy

#### Annotate Your Code:
# def add(a: int, b: int) -> int:
#     return a + b

#### Run Type Checker:
# mypy my_module.py

---

### 5️⃣ Configure Continuous Integration (CI) (Optional)
Use **GitHub Actions** to run tests, coverage, linting, and type checks on every pull request.

#### Create Workflow File: `.github/workflows/ci.yml`
# name: CI
#
# on: [push, pull_request]
#
# jobs:
#   build:
#     runs-on: ubuntu-latest
#
#     steps:
#     - uses: actions/checkout@v3
#
#     - name: Set up Python
#       uses: actions/setup-python@v4
#       with:
#         python-version: '3.x'
#
#     - name: Install dependencies
#       run: |
#         pip install -r requirements.txt
#         pip install pytest coverage flake8 black mypy
#
#     - name: Run tests with coverage
#       run: |
#         coverage run -m pytest
#         coverage report -m
#
#     - name: Run linter
#       run: flake8
#
#     - name: Run formatter check
#       run: black --check .
#
#     - name: Run type checks
#       run: mypy .

---

## ✅ Day 23 Checklist

| **Task**                                      | **Done?** |
|-----------------------------------------------|-----------|
| Installed and configured `pytest`             | ☐         |
| Wrote tests for core functionality            | ☐         |
| Measured code coverage with `coverage`        | ☐         |
| Integrated `black` for formatting             | ☐         |
| Integrated `flake8` for linting               | ☐         |
| Added type annotations and ran `mypy`         | ☐         |
| Set up GitHub Actions CI for tests and linting | ☐         |(Optional)

---

By completing these tasks, you’ll ensure your codebase is clean, maintainable, and ready for collaboration. 🚀