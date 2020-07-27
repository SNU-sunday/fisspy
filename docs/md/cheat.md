---
id: "my-style"
---

@import "fisspy_guide.less"

# IDL/Python Cheat Sheet

The primary useful scientific data analysis packages in Python are NumPy and SciPy. You import NumPy:

```Python
import numpy as np
```

Here you can compare the IDL and python operator.

**Relational Operators**

| IDL | Python |
|-----|--------|
| a EQ b | a == b |
| a NE b | a != b |
| a LT b | a < b |
| a LE b | a <= b |
| a GT b | a > b |
| a GE b | a >= b |

**Logical Operators**
| IDL | Python |
|-----|--------|
| a and b | a and b (np.logical_and(a, b))|
| a or b | a or b (np.logical_or(a, b))|

**MathFunctions**
| IDL | Python |
|-----|--------|
| sin(a) | np.sin(a) |
| alog(a) | np.log(a) |
| alog10(a) | np.log10(a) |
| exp(a) | np.exp(a) |

**Math Constants**
| IDL | Python |
|-----|--------|
| !pi | np.pi |

**Array Creation**
| IDL | Python |
|-----|--------|
| dblarr(3, 5) | np.zeros((3, 5)) |
| intarr(3, 5) | np.zeros((3, 5), dtype=int) |
| dblarr(3, 5)+1 | np.ones((3, 5)) |
| boolarr(10) | np.zeros(10, dtype=bool) |
| indgen(10) | np.arange(10) |
| dindgen(10) | np.arange(10, dtype=float) |

## More example
See this [page](http://mathesaurus.sourceforge.net/idl-numpy.html).
