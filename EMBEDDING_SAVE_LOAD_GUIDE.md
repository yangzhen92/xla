# PyTorch nn.Embedding Save and Load Guide

## Problem
Your `nn.Embedding` layer is not being saved in the model's `state_dict`, so when you load the model, the embedding weights are missing.

## Common Causes

1. **Embedding not registered as a module attribute**
   ```python
   # ❌ WRONG - embedding not in state_dict
   class Model(nn.Module):
       def __init__(self):
           super().__init__()
           embedding = nn.Embedding(1000, 128)  # Not registered!
           self.layers = [embedding]
   ```

2. **Embedding stored in regular Python list/dict**
   ```python
   # ❌ WRONG - regular dict doesn't register modules
   class Model(nn.Module):
       def __init__(self):
           super().__init__()
           self.embeddings = {'token': nn.Embedding(1000, 128)}  # Not registered!
   ```

## Solutions

### ✅ Solution 1: Register as Module Attribute (Recommended)
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)  # ✅ Registered!
        self.fc = nn.Linear(128, 256)
    
    def forward(self, x):
        return self.fc(self.embedding(x))

# Save and load normally
model = Model()
torch.save(model.state_dict(), 'model.pth')
new_model = Model()
new_model.load_state_dict(torch.load('model.pth'))
```

### ✅ Solution 2: Use add_module() for Dynamic Registration
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        embedding = nn.Embedding(1000, 128)
        self.add_module('embedding', embedding)  # ✅ Register it!
```

### ✅ Solution 3: Use nn.ModuleDict or nn.ModuleList
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ ModuleDict properly registers modules
        self.embeddings = nn.ModuleDict({
            'token': nn.Embedding(1000, 128)
        })
```

### ✅ Solution 4: Manual Save/Load (For Unregistered Embeddings)
```python
# Save
checkpoint = {
    'model_state_dict': model.state_dict(),
    'embedding_state_dict': embedding.state_dict(),
    'embedding_config': {
        'num_embeddings': embedding.num_embeddings,
        'embedding_dim': embedding.embedding_dim,
    }
}
torch.save(checkpoint, 'model.pth')

# Load
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
embedding = nn.Embedding(**checkpoint['embedding_config'])
embedding.load_state_dict(checkpoint['embedding_state_dict'])
```

## Verification

Always check if embedding is in state_dict:
```python
model = Model()
state_dict = model.state_dict()
print('embedding.weight' in state_dict)  # Should be True
print(list(state_dict.keys()))  # Should include embedding keys
```

## Quick Fix Checklist

- [ ] Is `self.embedding = nn.Embedding(...)` in `__init__`?
- [ ] Is embedding accessed via `self.embedding` (not a local variable)?
- [ ] If using collections, are you using `nn.ModuleDict` or `nn.ModuleList`?
- [ ] Check `model.state_dict().keys()` to verify embedding is included
- [ ] If embedding can't be registered, save/load it separately

## Example: Complete Working Code

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=128):
        super().__init__()
        # ✅ Correctly registered embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 256)
        self.output = nn.Linear(256, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return self.output(x)

# Usage
model = MyModel()
torch.save(model.state_dict(), 'model.pth')

# Load
new_model = MyModel()
new_model.load_state_dict(torch.load('model.pth'))
```
