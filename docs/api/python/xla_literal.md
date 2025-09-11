## Python API: `xla.python_api.xla_literal`

Conversion utilities between XLA `LiteralProto` and NumPy arrays.

### Functions
- `ConvertLiteralToNumpyArray(literal) -> np.ndarray | tuple`
  - Converts an XLA literal (including tuples) to NumPy.

- `ConvertNumpyArrayToLiteral(value) -> xla_data_pb2.LiteralProto`
  - Accepts a NumPy array or nested tuple of arrays.

### Examples
```python
import numpy as np
from xla.python_api import xla_literal

a = np.arange(6, dtype=np.float32).reshape(2, 3)
lit = xla_literal.ConvertNumpyArrayToLiteral(a)
back = xla_literal.ConvertLiteralToNumpyArray(lit)
assert np.allclose(a, back)
```

