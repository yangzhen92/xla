## Python API: `xla.python.xla_client`

Public classes, functions, and enums re-exported from `jax.jaxlib.xla_client` and local helpers.

### Types and Enums
- `PrimitiveType` — Enum of XLA primitive element types.
- `Shape` — Shape constructor from JAX's XLA client.
- `FftType`, `ShapeIndex` — Protocol buffer-backed enums.
- `ResultAccuracy`, `ResultAccuracyMode` — Controls numerical checking.
- `PrecisionConfig` — Controls operand precision.
- `OpMetadata` — Captures source information for ops.

### Functions
- `dtype_to_etype(dtype) -> PrimitiveType`
  - Convert a NumPy/`ml_dtypes` dtype to an XLA `PrimitiveType`.
  - Example:
    ```python
    from xla.python import xla_client as xc
    import numpy as np
    etype = xc.dtype_to_etype(np.float32)
    ```

- `current_source_info_metadata(op_type=None, op_name=None, skip_frames=1) -> OpMetadata`
  - Produce `OpMetadata` reflecting the current Python stack frame.

- `shape_from_pyval(pyval, layout: Sequence[int] | None = None) -> Shape`
  - Build an XLA `Shape` from a numpy array or tuple tree thereof.
  - Example:
    ```python
    import numpy as np
    from xla.python import xla_client as xc
    shp = xc.shape_from_pyval(np.zeros((2, 3), dtype=np.float32))
    ```

- `window_padding_type_to_pad_values(padding_type, lhs_dims, rhs_dims, window_strides) -> list[tuple[int,int]]`
  - Translate SAME/VALID to explicit padding pairs.

- `make_padding_config(padding_config | list[tuple[int,int,int]]) -> PaddingConfig`
- `make_dot_dimension_numbers(((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))) -> DotDimensionNumbers`
- `make_convolution_dimension_numbers(dimension_numbers | tuple[str,str,str], num_spatial_dimensions: int) -> ConvolutionDimensionNumbers`
- `make_replica_groups(replica_groups: Sequence[Sequence[int]] | None) -> list[ReplicaGroup]`

### Usage example
```python
import numpy as np
from xla.python import xla_client as xc

a = np.arange(6, dtype=np.float32).reshape(2, 3)
shape = xc.shape_from_pyval(a)
etype = xc.dtype_to_etype(a.dtype)
meta = xc.current_source_info_metadata(op_type="Test", op_name="demo")
```

Notes:
- This module primarily re-exports types from `jax.jaxlib.xla_client` and adds convenience helpers.

