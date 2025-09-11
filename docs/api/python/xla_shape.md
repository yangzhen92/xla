## Python API: `xla.python_api.xla_shape`

Utilities for constructing and inspecting XLA shapes.

### Classes
- `Shape(element_type, dimensions, layout=None)`
  - Wraps `xla_data_pb2.ShapeProto` with convenient accessors.
  - Methods: `element_type()`, `is_tuple()`, `dimensions()`, `tuple_shapes()`, `layout()`.

### Functions
- `Shape.from_pyval(pyval) -> Shape`
- `CreateShapeFromNumpy(value) -> Shape`
- `CreateShapeFromDtypeAndTuple(dtype, shape_tuple) -> Shape`

### Examples
```python
import numpy as np
from xla.python_api import xla_shape

arr = np.zeros((4, 5), dtype=np.float32)
shape = xla_shape.CreateShapeFromNumpy(arr)

tuple_shape = xla_shape.CreateShapeFromNumpy((arr, arr))
assert tuple_shape.is_tuple()
```

### Notes
- Layout is inferred from NumPy memory order (C vs Fortran) when possible.

