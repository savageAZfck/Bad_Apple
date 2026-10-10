fn main() {
    let path = std::env::args().nth(1).unwrap();
    let text = std::env::args().nth(2).unwrap();
    let t = tokenizers::Tokenizer::from_file(&path).unwrap();
    let e = t.encode(text.as_str(), true).unwrap();
    println!("ids: {:?}", e.get_ids());
    println!("toks: {:?}", e.get_tokens());
}
