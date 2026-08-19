//! Probe: dump every layer's Linear weight.len() vs in_features*out_features
//! to find the mismatch that panics ctx_dispatch_linear_cpu.
use pesti_runner::transformer::LlamaModel;

fn check(name: &str, wlen: usize, in_f: usize, out_f: usize) {
    let prod = in_f * out_f;
    let ok = wlen == prod;
    let flag = if ok { "" } else { "  <<< MISMATCH" };
    println!(
        "  {:14} wlen={:9}  in*out={:9} (in={:5} out={:5}){}",
        name, wlen, prod, in_f, out_f, flag
    );
}

fn main() {
    let path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );
    let model = LlamaModel::load_gguf(path).expect("load");
    println!("config: heads={} kv={} head_dim={} embed={} inter={} layers={}",
        model.config.num_heads, model.config.num_kv_heads, model.config.head_dim,
        model.config.embed_dim, model.config.intermediate_dim, model.config.num_layers);

    for (li, layer) in model.layers.iter().enumerate() {
        if li > 1 { break; } // just first two layers
        println!("\n=== layer {} ===", li);
        let a = &layer.attention;
        check("wq", a.wq.weight.len(), a.wq.in_features, a.wq.out_features);
        check("wk", a.wk.weight.len(), a.wk.in_features, a.wk.out_features);
        check("wv", a.wv.weight.len(), a.wv.in_features, a.wv.out_features);
        check("wo", a.wo.weight.len(), a.wo.in_features, a.wo.out_features);
        let f = &layer.feed_forward;
        check("w1", f.w1.weight.len(), f.w1.in_features, f.w1.out_features);
        check("w2", f.w2.weight.len(), f.w2.in_features, f.w2.out_features);
        check("w3", f.w3.weight.len(), f.w3.in_features, f.w3.out_features);
    }

    // Also dump the output head
    if let Some(out) = &model.output {
        println!("\n=== output head ===");
        check("lm_head", out.weight.len(), out.in_features, out.out_features);
    }
    if let Some(emb) = &model.token_embeddings {
        println!("\n=== token_embeddings ===");
        check("embed", emb.weight.len(), emb.in_features, emb.out_features);
    }
}
