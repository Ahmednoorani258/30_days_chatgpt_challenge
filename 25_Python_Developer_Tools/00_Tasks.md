# 🚀 Day 25: Master Advanced Python Developer Tools

## 🎯 Goal
Today, you’ll set up a professional development environment using advanced Python tools to increase productivity, manage complexity, and debug with confidence. You’ll learn how to structure real-world projects, use virtual environments, and master debugging, profiling, and packaging.

---

## 🧰 Tools You’ll Use

| **Category**           | **Tool(s)**                                |
|-------------------------|--------------------------------------------|
| **Virtual Environments** | `venv`, `pipenv`, or `poetry`             |
| **Debugging**           | `pdb`, `debugpy`, `breakpoint()`           |
| **Profiling**           | `cProfile`, `line_profiler`               |
| **Dependency Management** | `requirements.txt`, `pyproject.toml`     |
| **Packaging & Publishing** | `setuptools`, `build`, `twine`          |

---

## 🔧 Step-by-Step Tasks

### 1️⃣ Set Up a Virtual Environment
Best practice: Never develop directly in the global Python environment.

#### Using `venv` (built-in):
- `# python -m venv venv`
- `# source venv/bin/activate` (macOS/Linux)
- `# venv\Scripts\activate` (Windows)

#### Optional: Use `pipenv` or `poetry` for more features:
- `# pip install pipenv`
- `# pipenv install`

---

### 2️⃣ Project Structure for Maintainability
Organize your code like a pro:


arduino
Copy code
my_project/
├── src/
│   └── my_module/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   └── test_core.py
├── README.md
├── pyproject.toml
├── requirements.txt
└── setup.py
3️⃣ Add Debugging Power
Use breakpoints in scripts:
### 3️⃣ Add Debugging Power
Use breakpoints in scripts:
- `# def buggy_func():`
- `#     a = 10`
- `#     breakpoint()  # built-in debugger`
- `#     b = a / 0`
- `# buggy_func()`

#### For full debugging:
Install VS Code Debugger or `debugpy`:
- `# pip install debugpy`

---

### 4️⃣ Profile and Optimize Your Code
Use the built-in profiler:
- `# python -m cProfile my_script.py`

Install `line_profiler` for per-line performance:
- `# pip install line_profiler`

Add `@profile` decorator to functions, then run:
- `# kernprof -l -v my_script.py`

---

### 5️⃣ Create a Package (Local or PyPI Ready)
Create `setup.py`:
- `# from setuptools import setup, find_packages`
- `# setup(`
- `#     name="my_module",`
- `#     version="0.1.0",`
- `#     packages=find_packages(where="src"),`
- `#     package_dir={"": "src"},`
- `# )`

Build and install locally:
- `# pip install -e .`

Optionally build for PyPI:
- `# pip install build twine`
- `# python -m build`
- `# twine check dist/*`

---

### 6️⃣ Write a `pyproject.toml` (Modern Projects)
- `# [build-system]`
- `# requires = ["setuptools>=42"]`
- `# build-backend = "setuptools.build_meta"`

- `# [project]`
- `# name = "my-module"`
- `# version = "0.1.0"`
- `# description = "My ML Model as a Python Package"`

---

### 7️⃣ Add `.env` for Secrets and Config
Install `python-dotenv`:
- `# pip install python-dotenv`

Create `.env`:
- `# API_KEY=your_api_key`

Load it in code:
- `# from dotenv import load_dotenv`
- `# import os`
- `# load_dotenv()`
- `# api_key = os.getenv("API_KEY")`

---

## ✅ Day 25 Checklist

| **Task**                                      | **Done?** |
|-----------------------------------------------|-----------|
| Created and activated virtual environment     | ☐         |
| Structured a professional Python project      | ☐         |
| Used `breakpoint()` or `pdb` for debugging    | ☐         |
| Profiled code performance with `cProfile`     | ☐         |
| Created a `setup.py` for packaging            | ☐         |
| Built and optionally installed your package   | ☐         |
| Used `.env` with `dotenv` for secrets         | ☐         |

---

## 💡 Pro Tip:
Start treating your projects like production code—even your experiments. This builds great habits and opens doors to DevOps, MLOps, and more!