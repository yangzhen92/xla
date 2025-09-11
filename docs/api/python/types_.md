## Python API: `xla.python_api.types_`

Type conversion records and mappings between XLA primitive types and NumPy/`ml_dtypes`.

### Data structures
- `TypeConversionRecord(primitive_type, numpy_dtype, literal_field_name, literal_field_type)`

### Mappings
- `MAP_XLA_TYPE_TO_RECORD: dict[PrimitiveType, TypeConversionRecord]`
- `MAP_DTYPE_TO_RECORD: dict[str, TypeConversionRecord]`

### Example
```python
import numpy as np
from xla.python_api import types_

record = types_.MAP_DTYPE_TO_RECORD[str(np.dtype('float32'))]
assert record.literal_field_name == 'f32s'
```

