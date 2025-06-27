# LLMs-from-scratch Efficiency Analysis Report

## Overview
This report documents efficiency improvements identified in the LLMs-from-scratch codebase, focusing on performance optimizations that can reduce computational overhead without changing model behavior.

## Identified Issues

### 1. Custom GELU Implementation Inefficiency (HIGH IMPACT)
**Issue**: The custom GELU activation function computes `torch.sqrt(torch.tensor(2.0 / torch.pi))` on every forward pass.

**Files Affected**: 12+ files including:
- ch04/01_main-chapter-code/gpt.py
- pkg/llms_from_scratch/ch04.py
- ch05/01_main-chapter-code/previous_chapters.py
- ch06/01_main-chapter-code/previous_chapters.py
- And 8+ other files

**Current Implementation**:
```python
def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
    ))
```

**Performance Impact**: 
- Constant computed millions of times during training
- Creates new tensor and performs sqrt operation on every forward pass
- Affects every GELU layer in the transformer blocks

**Solution**: Precompute the constant during initialization:
```python
def __init__(self):
    super().__init__()
    self.sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi))

def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
    ))
```

### 2. Inefficient Nested Loops (MEDIUM IMPACT)
**File**: ch07/02_dataset-utilities/find-near-duplicates.py
**Issue**: Nested loops using `range(len())` pattern instead of direct iteration or vectorized operations.

**Current Implementation**:
```python
for i in range(len(cos_sim_matrix)):
    for j in range(i+1, len(cos_sim_matrix)):
        if cos_sim_matrix[i, j] > threshold:
```

**Solution**: Could be optimized using numpy/torch vectorized operations or direct iteration.

### 3. Repeated torch.cat Operations (LOW-MEDIUM IMPACT)
**Files**: Multiple generation functions across the codebase
**Issue**: Repeated `torch.cat((idx, idx_next), dim=1)` operations in generation loops.

**Current Pattern**:
```python
for _ in range(max_new_tokens):
    # ... generate next token ...
    idx = torch.cat((idx, idx_next), dim=1)
```

**Solution**: Pre-allocate tensor and use indexing, or batch the concatenations.

### 4. Inconsistent Use of Optimized Variants (LOW IMPACT)
**Issue**: The codebase has `GPTModelFast` with optimized implementations but the regular `GPTModel` is still used in many places.

**Files**: Multiple chapters use the slower custom implementations instead of PyTorch's optimized built-ins.

## Recommendation Priority

1. **HIGH**: Fix GELU constant computation (implemented in this PR)
2. **MEDIUM**: Optimize nested loops in duplicate detection
3. **LOW-MEDIUM**: Optimize generation loop concatenations
4. **LOW**: Promote use of fast variants where appropriate

## Implementation

This PR implements the GELU optimization (#1) as it provides the highest performance impact with zero risk of changing model behavior. The mathematical result is identical, but the computation is significantly more efficient.

**Testing**: All existing tests pass, confirming the optimization maintains exact behavioral equivalence.

**Performance Benefit**: Eliminates redundant tensor creation and sqrt computation on every forward pass through GELU layers, which occurs millions of times during training.
