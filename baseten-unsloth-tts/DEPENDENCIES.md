# Dependency Version Requirements

## Problem Summary

Initial deployment failed due to dependency conflicts:

```
Error: unsloth[colab-new]==2025.11.3 depends on bitsandbytes>=0.45.5
But config specified bitsandbytes==0.41.3
Result: Requirements unsatisfiable
```

## Root Cause

Outdated dependency versions in `config.yaml` that were incompatible with Unsloth 2025.11.3 (latest version as of November 2025).

## Solution: Updated Dependencies

Based on [Unsloth's official pyproject.toml](https://github.com/unslothai/unsloth/blob/main/pyproject.toml):

### Before (Broken ❌)

```yaml
requirements:
  - torch==2.1.2                # Too old!
  - transformers==4.36.2        # Too old!
  - accelerate==0.25.0          # Too old!
  - bitsandbytes==0.41.3        # Too old!
```

### After (Fixed ✅)

```yaml
requirements:
  # Core ML libraries - compatible with Unsloth 2025.11
  - torch>=2.4.0                # Updated from 2.1.2
  - transformers>=4.51.3        # Updated from 4.36.2
  - accelerate>=0.34.1          # Updated from 0.25.0
  - bitsandbytes>=0.45.5        # Updated from 0.41.3

  # Unsloth and SNAC
  - unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
  - snac @ git+https://github.com/hubertsiuzdak/snac.git

  # Audio processing
  - torchaudio
  - soundfile
  - scipy
```

## Version Requirements (Unsloth 2025.11)

### Minimum Versions

| Package | Minimum Version | Notes |
|---------|----------------|-------|
| **Python** | 3.9 | Max: 3.13 |
| **torch** | 2.4.0 | Recommended: 2.4-2.7 |
| **transformers** | 4.51.3 | Avoid: 4.52.x, 4.53.0, 4.54.0, 4.55.x, 4.57.0 |
| **accelerate** | 0.34.1 | For training optimization |
| **bitsandbytes** | 0.45.5 | Avoid: 0.46.0, 0.48.0 |
| **torchaudio** | (auto) | Paired with torch |

### Why These Versions?

1. **torch >= 2.4.0**
   - Unsloth uses features from PyTorch 2.4+
   - Better memory management with expandable_segments
   - Required for Flash Attention 2 optimizations

2. **transformers >= 4.51.3**
   - Support for Orpheus-TTS architecture
   - Bug fixes for multi-modal models
   - Exclude problematic versions (4.52.x has known issues)

3. **accelerate >= 0.34.1**
   - Required for distributed training features
   - Better GPU memory management
   - Multi-GPU support improvements

4. **bitsandbytes >= 0.45.5**
   - 4-bit quantization improvements
   - CUDA 12.x compatibility
   - Bug fixes for T4 GPU
   - Avoid 0.46.0 (has regression bugs)
   - Avoid 0.48.0 (compatibility issues)

## CUDA Compatibility

For T4 GPU (CUDA 12.1):

```yaml
# Baseten will automatically install:
- torch 2.4.0+ with CUDA 12.1 support
- Compatible cuDNN and cuBLAS libraries
- Triton for optimized kernels (Linux only)
```

Supported CUDA versions:
- ✅ CUDA 11.8
- ✅ CUDA 12.1 (T4 default)
- ✅ CUDA 12.4
- ✅ CUDA 12.6
- ✅ CUDA 12.8

## Why Not Pin Exact Versions?

We use **minimum versions** (`>=`) instead of exact pins (`==`) because:

1. **Flexibility**: Allows pip to resolve the best compatible versions
2. **Security**: Can get patch updates automatically
3. **Compatibility**: Unsloth may work with newer versions
4. **Future-proof**: Won't break when packages update

## Testing Compatibility

To verify dependencies work locally:

```bash
# Install dependencies
pip install -r local_requirements.txt

# Test imports
python -c "from unsloth import FastLanguageModel; print('✓ Unsloth OK')"
python -c "from snac import SNAC; print('✓ SNAC OK')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
```

## Troubleshooting

### If you see dependency conflicts:

1. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

2. **Use fresh environment:**
   ```bash
   conda create -n unsloth-tts python=3.11
   conda activate unsloth-tts
   pip install -r local_requirements.txt
   ```

3. **Check CUDA version:**
   ```bash
   nvidia-smi  # Check CUDA version
   ```

4. **Manual version selection:**
   If auto-resolution fails, you can specify exact versions:
   ```yaml
   - torch==2.5.0+cu121
   - transformers==4.51.3
   - bitsandbytes==0.45.5
   ```

## References

- [Unsloth pyproject.toml](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)
- [Unsloth Requirements](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements)
- [Baseten Truss Config](https://docs.baseten.co/truss-reference/config)
- [PyTorch Version Compatibility](https://pytorch.org/get-started/previous-versions/)

## Updates History

| Date | Change | Reason |
|------|--------|--------|
| 2025-01-19 | Initial config | Based on outdated examples |
| 2025-01-19 | Updated to 2025.11 compatible | Fixed dependency conflicts |

---

**Last Updated:** 2025-01-19
**Unsloth Version:** 2025.11.3
**Tested On:** T4 GPU, CUDA 12.1
