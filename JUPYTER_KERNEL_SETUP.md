# Jupyter Kernel Setup Guide

## Which Kernel to Use

**Answer: Use the Python 3 kernel (IPython kernel)**

This project is a Python-based machine learning project that requires:
- Python 3.8 or higher
- IPython kernel for Jupyter notebooks
- Dependencies listed in `requirements.txt`

## Quick Start

### 1. Check if Jupyter is installed
```bash
jupyter --version
```

If not installed, run:
```bash
pip install jupyter
```

### 2. List available kernels
```bash
jupyter kernelspec list
```

You should see something like:
```
Available kernels:
  python3    /usr/local/share/jupyter/kernels/python3
```

### 3. Verify Python version
```bash
python3 --version
```
Ensure it's Python 3.8 or higher.

## Setting Up a Virtual Environment Kernel

### Using venv

**Step 1: Create virtual environment**
```bash
cd bank-marketing-classification
python3 -m venv venv
```

**Step 2: Activate environment**
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Install IPython kernel**
```bash
pip install ipykernel
```

**Step 5: Add environment as Jupyter kernel**
```bash
python -m ipykernel install --user --name=bank-marketing --display-name="Python (bank-marketing)"
```

### Using Conda

**Step 1: Create conda environment**
```bash
conda create -n bank-marketing python=3.8
```

**Step 2: Activate environment**
```bash
conda activate bank-marketing
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Install IPython kernel**
```bash
conda install ipykernel
```

**Step 5: Add environment as Jupyter kernel**
```bash
python -m ipykernel install --user --name=bank-marketing --display-name="Python (bank-marketing)"
```

## Selecting the Kernel in Jupyter

### In Jupyter Notebook
1. Open `prompt_III.ipynb`
2. Click **Kernel** → **Change Kernel**
3. Select **Python 3** or **Python (bank-marketing)** (if you set up a custom kernel)

### In Jupyter Lab
1. Open `prompt_III.ipynb`
2. Click the kernel name in the top-right corner (e.g., "Python 3")
3. Select your desired kernel from the dropdown

### In VS Code
1. Open `prompt_III.ipynb`
2. Click **Select Kernel** in the top-right
3. Choose **Python Environments** → **Python (bank-marketing)** or **Python 3**

## Verifying the Kernel Setup

After opening the notebook with your kernel, run this in a cell:

```python
import sys
import pandas as pd
import numpy as np
import sklearn

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
```

Expected output (versions may vary):
```
Python version: 3.8.x (or higher)
Python executable: /path/to/your/venv/bin/python
Pandas version: 1.5.3
NumPy version: 1.24.3
Scikit-learn version: 1.2.2
```

## Managing Jupyter Kernels

### List all kernels
```bash
jupyter kernelspec list
```

### Remove a kernel
```bash
jupyter kernelspec remove <kernel-name>

# Example:
jupyter kernelspec remove bank-marketing
```

### Rename a kernel
You can't rename directly, but you can:
1. Remove the old kernel
2. Reinstall with the new name

```bash
jupyter kernelspec remove old-name
python -m ipykernel install --user --name=new-name --display-name="New Display Name"
```

## Kernel Specifications

The Python 3 kernel uses a `kernel.json` specification file. You can find it at:
```
~/.local/share/jupyter/kernels/python3/kernel.json
```

Example kernel.json:
```json
{
  "argv": [
    "/usr/bin/python3",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
```

## Common Issues and Solutions

### Issue 1: "No module named 'ipykernel'"
**Solution:**
```bash
pip install ipykernel
```

### Issue 2: Kernel not showing in Jupyter
**Solution:**
```bash
# Reinstall the kernel
python -m ipykernel install --user --name=bank-marketing --display-name="Python (bank-marketing)"

# Restart Jupyter
```

### Issue 3: Wrong Python version in kernel
**Solution:**
Check which Python is being used:
```bash
which python
python --version
```
Make sure you're in the correct virtual environment.

### Issue 4: Packages not found in notebook
**Solution:**
The notebook might be using a different Python than your terminal. Check in a notebook cell:
```python
import sys
print(sys.executable)
```

If it's wrong, create a new kernel pointing to the correct Python:
```bash
/path/to/correct/python -m ipykernel install --user --name=correct-env
```

### Issue 5: Kernel keeps dying
**Possible causes:**
- Insufficient memory
- Conflicting packages
- Corrupted installation

**Solutions:**
```bash
# Restart Jupyter server
# Clear all outputs: Cell → All Output → Clear
# Restart kernel: Kernel → Restart & Clear Output

# Reinstall Jupyter and ipykernel
pip uninstall jupyter ipykernel
pip install jupyter ipykernel
```

## Best Practices

1. **Use virtual environments**: Isolate project dependencies
2. **One kernel per project**: Create dedicated kernels for each project
3. **Descriptive names**: Use clear display names like "Python (bank-marketing)"
4. **Keep dependencies updated**: Regularly update packages for security and features
5. **Document your setup**: Note which kernel and Python version you're using

## Alternative Kernels (Not Required for This Project)

While this project uses Python 3, Jupyter supports many kernels:
- **R**: For R language
- **Julia**: For Julia language
- **JavaScript**: For Node.js
- **Scala**: For Apache Spark
- **Ruby**: For Ruby language

You can find more at: https://github.com/jupyter/jupyter/wiki/Jupyter-kernels

## Additional Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [IPython Kernel Documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
- [Jupyter Kernels Wiki](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)
- [Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Conda Documentation](https://docs.conda.io/en/latest/)

## Quick Reference Commands

```bash
# Install Jupyter
pip install jupyter

# Install IPython kernel
pip install ipykernel

# List kernels
jupyter kernelspec list

# Add new kernel
python -m ipykernel install --user --name=mykernel --display-name="My Kernel"

# Remove kernel
jupyter kernelspec remove mykernel

# Start Jupyter Notebook
jupyter notebook

# Start Jupyter Lab
jupyter lab

# Check Python version in terminal
python --version

# Check Python version in notebook cell
import sys; print(sys.version)
```

---

**For this project, use: Python 3 kernel with Python 3.8+**

If you encounter any issues, refer to the troubleshooting section or open an issue on GitHub.
