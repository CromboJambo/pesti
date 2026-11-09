// Fused attention kernel for autoregressive decode (batch=1, seq=1).
// Computes: softmax(q @ K^T / sqrt(d)) @ V in a single kernel launch.
//
// q:        [head_dim]         - query vector (RoPE applied)
// k_cache:  [seq_len, head_dim] - cached keys (RoPE applied)
// v_cache:  [seq_len, head_dim] - cached values
// seq_len:  number of cached positions
// head_dim: dimension of each head
// scale:    1/sqrt(head_dim)
// output:   [head_dim]         - attention output for this head

extern "C" __global__ void fused_attention_kernel(
    const float* q,           // [head_dim]
    const float* k_cache,     // [seq_len * head_dim]
    const float* v_cache,     // [seq_len * head_dim]
    float scale,
    int seq_len,
    int head_dim,
    float* output)            // [head_dim]
{
    // Each thread computes one output dimension
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    // Compute attention scores for this position against all cached positions
    // Scores computed in registers (seq_len is small enough for decode)
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // First pass: compute scores and find max for numerically stable softmax
    for (int pos = 0; pos < seq_len; pos++) {
        float score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            score += q[j] * k_cache[pos * head_dim + j];
        }
        score *= scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    // Second pass: compute softmax weights and weighted sum of V
    float out_val = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
        float score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            score += q[j] * k_cache[pos * head_dim + j];
        }
        score *= scale;

        // Softmax weight
        float w = expf(score - max_score);
        sum_exp += w;
        out_val += w * v_cache[pos * head_dim + d];
    }

    // Normalize by sum of exps
    output[d] = out_val / sum_exp;
}
