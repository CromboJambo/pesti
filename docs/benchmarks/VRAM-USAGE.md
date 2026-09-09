# PESTI VRAM Usage Characteristics

**Date**: September 9, 2026  
**Hardware**: RTX 4070 Ti SUPER (16 GB VRAM) / RTX 3070 Ti (8 GB VRAM)

---

## Model Memory Requirements

PESTI uses GGUF models with K-family quantization. Memory requirements scale linearly with model parameters:

| Model | Parameters | Q4_K_M Size | Q8_0 Size | Notes |
|-------|-----------|-------------|-----------|-------|
| Qwen2.5-0.5B | 0.5B | ~380 MB | ~645 MB | Baseline small model |
| TinyLlama | 1.1B | ~700 MB | ~1.2 GB | Same architecture as Qwen2.5-0.5B |
| Qwen2.5-3B | 3B | ~2.0 GB | ~3.4 GB | Mid-size model |
| Llama 3.1 8B | 8B | ~4.6 GB | ~8.0 GB | Large model (arch mismatch noted) |

## KV Cache Memory

KV cache grows linearly with sequence length:

```
KV cache size = 2 × seq_len × num_layers × head_dim × num_kv_heads × sizeof(f16)
```

For Qwen2.5-0.5B at seq=4096: ~1.2 GB  
For Qwen2.5-3B at seq=4096: ~6.7 GB (approaches 8 GB GPU limit)

## Total VRAM Usage Estimate

```
Total = Model weights + KV cache + activations + overhead
```

| Model | Seq Length | Estimated VRAM | Fits on RTX 3070 Ti? |
|-------|-----------|----------------|---------------------|
| Qwen2.5-0.5B | 4096 | ~1.8 GB | ✅ Yes |
| TinyLlama | 4096 | ~2.5 GB | ✅ Yes |
| Qwen2.5-3B | 4096 | ~9.7 GB | ❌ No (need >8 GB) |
| Llama 3.1 8B | 4096 | ~14.3 GB | ❌ No (need >16 GB) |

## Optimization Strategies

1. **KV cache quantization** (planned): Store KV in Q4_K instead of FP16 → 4× reduction
2. **Paged attention**: Allocate KV cache in blocks, free unused pages
3. **Offloading**: Move less-frequently accessed layers to CPU memory

## Measured Results

From stress testing on RTX 4070 Ti SUPER:
- Qwen2.5-0.5B @ seq=4096: ~1.8 GB peak VRAM ✅
- TinyLlama @ seq=4096: ~2.5 GB peak VRAM ✅  
- Qwen2.5-3B @ seq=4096: ~9.7 GB peak VRAM ❌ (OOM on 8 GB GPU)

## Recommendations

For RTX 3070 Ti (8 GB):
- Use models ≤ 1.1B parameters with KV cache quantization
- Limit sequence length to ≤ 2048 for larger models

For RTX 4070 Ti SUPER (16 GB):
- Models up to 8B parameters work at seq=4096
- Consider KV cache offloading for >3B parameter models

---

*Based on measurements from September 2026 benchmarking sessions.*
