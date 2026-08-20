use bad_apple::tensor_brain::{text_to_grounded_embedding, BRAIN_DIM, CandleBrain};

fn main() {
    let brain = CandleBrain::new("diag", 2, &bad_apple::tensor_brain::layer_dims()).unwrap();
    let axes = [0.0; 4];
    for text in [
        "what time is it",
        "open workspace bad apple",
        "check the time corazon",
        "what is firefly inferno",
        "how do i implement semantic routing",
        "what is the meaning of this project",
    ] {
        let emb = text_to_grounded_embedding(text, &axes);
        let state = brain.forward(&emb).unwrap();
        let mean: f64 = state.iter().sum::<f64>() / BRAIN_DIM as f64;
        let var: f64 = state.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / BRAIN_DIM as f64;
        let norm: f64 = state.iter().map(|v| v * v).sum::<f64>().sqrt();
        println!("{text}: mean={mean:.4} var={var:.4} norm={norm:.4}");
    }
}
