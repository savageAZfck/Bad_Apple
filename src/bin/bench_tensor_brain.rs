use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    let brain = bad_apple::tensor_brain::CandleBrain::new(
        "gatekeeper-bench",
        2,
        &bad_apple::tensor_brain::layer_dims(),
    )?;

    let prompts = vec![
        "open workspace bad apple",
        "what time is it corazon",
        "bad apple explain the rust module",
        "how do i implement semantic routing in my ai os",
        "mi amor check the time",
    ];

    let axes = [0.0, 0.0, 0.0, 0.0];
    let mut out = vec![0.0; bad_apple::tensor_brain::BRAIN_DIM];

    // Warmup
    for prompt in &prompts {
        let embedding = bad_apple::tensor_brain::text_to_grounded_embedding(prompt, &axes);
        brain.forward_into(&embedding, &mut out)?;
    }

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        for prompt in &prompts {
            let embedding = bad_apple::tensor_brain::text_to_grounded_embedding(prompt, &axes);
            brain.forward_into(&embedding, &mut out)?;
        }
    }
    let elapsed = start.elapsed();
    let total = iters * prompts.len();
    let tok_per_sec = total as f64 / elapsed.as_secs_f64();
    let us_per_call = elapsed.as_micros() as f64 / total as f64;

    println!("576-D tensor_brain benchmark");
    println!("  iterations: {iters}");
    println!("  prompts: {}", prompts.len());
    println!("  total inferences: {total}");
    println!("  elapsed: {:.3}s", elapsed.as_secs_f64());
    println!("  throughput: {:.1} inferences/s", tok_per_sec);
    println!("  latency: {:.2} us/inference", us_per_call);

    Ok(())
}
