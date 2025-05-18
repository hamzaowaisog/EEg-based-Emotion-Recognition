# Model Loading Issue and Solutions

## The Issue

When loading model checkpoints with `weights_only=True`, you may encounter the following error:

```
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([scalar])` to allowlist this global if you trust this class/function.
```

This happens because PyTorch's secure loading mechanism with `weights_only=True` doesn't allow certain types by default, including NumPy's `numpy.core.multiarray.scalar` type.

## Solutions

### Solution 1: Remove `weights_only=True`

The simplest solution is to remove the `weights_only=True` parameter when loading the model:

```python
# Before
checkpoint = torch.load(os.path.join(output_dir, "best_model_acc.pth"), weights_only=True)

# After
checkpoint = torch.load(os.path.join(output_dir, "best_model_acc.pth"))
```

This is safe if you trust the source of the checkpoint (which you should if you created it yourself).

### Solution 2: Add NumPy scalar to safe globals

A more robust solution is to add the NumPy scalar type to PyTorch's safe globals list:

```python
import torch
import numpy as np
from torch.serialization import add_safe_globals

# Get the scalar type from numpy
scalar = np.array(0).item().__class__

# Add it to PyTorch's safe globals list
add_safe_globals([scalar])

# Now you can safely load with weights_only=True
checkpoint = torch.load(path, weights_only=True)
```

### Using the provided utility

We've created a utility module `safe_load.py` that implements Solution 2:

```python
from safe_load import safe_load

# Load the model safely with weights_only=True
checkpoint = safe_load("path/to/checkpoint.pth")

# Or specify a device
checkpoint = safe_load("path/to/checkpoint.pth", device=torch.device("cuda"))
```

## Why use `weights_only=True`?

The `weights_only=True` parameter is a security feature that prevents arbitrary code execution when loading models from untrusted sources. It's a good practice to use it when loading models from external sources, but it's not necessary when loading models you've created yourself.

## References

- [PyTorch Documentation on torch.load](https://pytorch.org/docs/stable/generated/torch.load.html)
- [PyTorch Security Best Practices](https://pytorch.org/docs/stable/notes/security.html)
