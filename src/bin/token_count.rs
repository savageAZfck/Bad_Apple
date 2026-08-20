use bad_apple::tensor_brain::tokenize_text;
fn main() {
    for text in ["what time is it", "open workspace bad apple", "check the time corazon", "what is firefly inferno", "how do i implement semantic routing", "what is the meaning of this project"] {
        println!("{}: {}", text, tokenize_text(text).len());
    }
}
