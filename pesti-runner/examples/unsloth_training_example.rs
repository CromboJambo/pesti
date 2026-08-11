//! Example: Unsloth-style efficient training with gradient checkpointing and Flash Attention

use pesti_runner::unsloth::{
    CheckpointedLayer, FlashAttentionConfig, MemoryEfficientLoRA, UnslothConfig,
};

fn main() {
    println!("=== Unsloth-Style Training Example ===\n");

    // 1. Configure unsloth optimizations
    let unsloth_config = UnslothConfig::default()
        .with_gradient_checkpointing(true)
        .with_flash_attention(true)
        .with_checkpoint_layers(4)
        .with_memory_efficient(true);

    println!("✓ Unsloth config created:");
    println!(
        "  - Gradient checkpointing: {}",
        unsloth_config.gradient_checkpointing
    );
    println!("  - Flash attention: {}", unsloth_config.flash_attention);
    println!(
        "  - Checkpoint layers: {}",
        unsloth_config.checkpoint_layers
    );
    println!("  - Memory efficient: {}", unsloth_config.memory_efficient);

    // 2. Create memory-efficient LoRA adapter
    let lora = MemoryEfficientLoRA::new(512, 1024, 8, 2.0).expect("Failed to create LoRA");

    println!("\n✓ Memory-efficient LoRA created:");
    println!("  - Input features: 512");
    println!("  - Output features: 1024");
    println!("  - Rank: {}", lora.rank);
    println!("  - Scaling: {}", lora.scaling);
    println!("  - LoRA A size: {} f32s", lora.lora_a.len());
    println!("  - LoRA B size: {} f32s", lora.lora_b.len());

    // 3. Create checkpointed transformer layers
    let num_layers = 12;
    let mut checkpointed_layers: Vec<CheckpointedLayer> = (0..num_layers)
        .map(|i| {
            CheckpointedLayer::new(
                i,
                i % unsloth_config.checkpoint_layers == 0, // Checkpoint every Nth layer
            )
        })
        .collect();

    println!("\n✓ Created {} checkpointed transformer layers", num_layers);
    println!(
        "  - Layers to checkpoint: {}",
        checkpointed_layers
            .iter()
            .filter(|l| l.should_checkpoint)
            .count()
    );

    // 4. Simulate forward pass with checkpointing
    let hidden_dim = 512;
    let input = vec![1.0f32; hidden_dim];

    for (i, layer) in checkpointed_layers.iter_mut().enumerate() {
        let output = layer.forward(&input, |x| x.to_vec());

        if i % 4 == 0 {
            println!(
                "  ✓ Layer {} forward pass (checkpointed): output size {}",
                i,
                output.len()
            );
        } else {
            println!("  ✓ Layer {} forward pass (no checkpoint)", i);
        }
    }

    // 5. Configure Flash Attention
    let flash_config = FlashAttentionConfig::new(8, 64, 2048)
        .with_causal(true)
        .with_softmax_scale(0.125);

    println!("\n✓ Flash Attention 2 config:");
    println!("  - Num heads: {}", flash_config.num_heads);
    println!("  - Head dim: {}", flash_config.head_dim);
    println!("  - Seq len: {}", flash_config.seq_len);
    println!("  - Causal mask: {}", flash_config.causal);
    println!("  - Softmax scale: {}", flash_config.softmax_scale);

    // 6. Calculate memory savings
    let batch_size = 1;
    let seq_len = flash_config.seq_len;
    let num_heads = flash_config.num_heads;
    let head_dim = flash_config.head_dim;

    // Standard attention: O(seq_len² * num_heads * head_dim)
    let standard_memory = (seq_len * seq_len * num_heads * head_dim) as f64;

    // Flash attention: O(seq_len * num_heads * head_dim)
    let flash_memory = (seq_len * num_heads * head_dim) as f64;

    println!(
        "\n✓ Memory comparison for batch_size={}, seq_len={}",
        batch_size, seq_len
    );
    println!(
        "  - Standard attention: {:.2} MB",
        standard_memory / (1024.0 * 1024.0)
    );
    println!(
        "  - Flash Attention 2: {:.2} MB",
        flash_memory / (1024.0 * 1024.0)
    );
    println!("  - Memory saved: {:.1}x", standard_memory / flash_memory);

    println!("\n=== Unsloth-Style Training Complete ===");
}
