//! Example: Using TRL-like training orchestrator with PESTI

use pesti_runner::trl::{Callbacks, CheckpointCallback, LoggingCallback, Metrics, State, Trainer, TrainingConfig};
use pesti_runner::peft::{Adapter, AdapterConfig, LoRAAdapter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TRL-like Training Example ===\n");

    // 1. Create training configuration using builder
    let config = TrainingConfig::default()
        .with_num_epochs(3)
        .with_batch_size(4)
        .with_learning_rate(2e-5);

    println!("Training config:");
    println!("  Epochs: {}", config.num_epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {:.2e}", config.learning_rate);
    println!();

    // 2. Create LoRA adapter (placeholder - in real usage, would load from model)
    let adapter_config = AdapterConfig::lora(8, 16.0);
    let adapter = LoRAAdapter::new_zeros(512, 1024, &adapter_config)?;

    println!("Adapter:");
    println!("  Rank: {}", adapter.rank());
    println!("  Scaling: {}", adapter.scaling());
    println!();

    // 3. Create trainer with builder pattern (placeholder model)
    let mut trainer = Trainer::new(
        /* model placeholder */ pesti_runner::transformer_stub::LlamaModel::default(), 
        adapter, 
        config.clone(),
        pesti_runner::trl::OptimizerConfig::default(),
    );

    // 4. Add callbacks
    let mut callbacks = Callbacks::new();
    callbacks.add(CheckpointCallback::new("checkpoints", 50));
    callbacks.add(LoggingCallback::new("training.log"));

    // 5. Simulate training loop (placeholder)
    println!("Starting training simulation...\n");

    for epoch in 0..config.num_epochs {
        trainer.state_mut().epoch = epoch;
        
        // Simulate loss values
        let losses: Vec<f32> = (1..=10).map(|i| 2.0 / (i as f32)).collect();
        
        for (step, loss) in losses.iter().enumerate() {
            trainer.state_mut().global_step = epoch * 10 + step;
            trainer.state_mut().record_loss(*loss);
            println!("Epoch {}: step {}, loss: {:.4}", epoch, step, loss);
        }

        // Simulate evaluation
        let eval_loss = losses.last().unwrap() / 2.0;
        let metrics = Metrics::new(eval_loss).with_perplexity((-eval_loss).exp());
        trainer.state_mut().record_eval(metrics.clone());
        
        println!(
            "Epoch {} complete - Eval loss: {:.4}, Perplexity: {:.2}\n", 
            epoch, eval_loss, metrics.perplexity.unwrap()
        );
    }

    // 6. Final state
    println!("Training complete!");
    println!("Final avg loss: {:.4}", trainer.state().avg_loss().unwrap());
    
    if let Some(best) = trainer.state().best_eval() {
        println!("Best eval loss: {:.4}", best.loss);
        println!(
            "Best perplexity: {:.2}", 
            best.perplexity.unwrap_or(f32::INFINITY)
        );
    }

    Ok(())
}
