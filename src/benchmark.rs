//! Simple external task benchmark for the Firefly agent.
//!
//! The suite generates math / logic / code puzzles, lets the agent produce a
//! Python solution, and records the success rate over time.

#[derive(Clone)]
pub struct BenchmarkTask {
    pub name: &'static str,
    pub prompt: &'static str,
    pub expected: &'static str,
    pub tolerance: Option<f64>,
}

pub struct BenchmarkSuite {
    pub tasks: Vec<BenchmarkTask>,
    pub history: Vec<(String, bool)>,
    pub index: usize,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            tasks: vec![
                BenchmarkTask {
                    name: "simple_arithmetic",
                    prompt: "Write a short Python 3 script that computes (17 * 23) + (12 * 31) and prints only the final integer answer.",
                    expected: "763",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "factorial",
                    prompt: "Write a short Python 3 script that computes 6! (6 factorial) and prints only the final integer answer.",
                    expected: "720",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "fibonacci_10",
                    prompt: "Write a short Python 3 script that computes the 10th Fibonacci number (F(0)=0, F(1)=1, F(2)=1, ... F(10)=?) and prints only the final integer answer.",
                    expected: "55",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "sum_of_squares",
                    prompt: "Write a short Python 3 script that computes 1*1 + 2*2 + 3*3 + 4*4 + 5*5 and prints only the final integer answer.",
                    expected: "55",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "prime_check",
                    prompt: "Write a short Python 3 script that checks if 29 is prime and prints only the boolean True or False.",
                    expected: "True",
                    tolerance: None,
                },
            ],
            history: Vec::new(),
            index: 0,
        }
    }

    pub fn next_task(&mut self) -> &BenchmarkTask {
        let task = &self.tasks[self.index % self.tasks.len()];
        self.index += 1;
        task
    }

    pub fn record(&mut self, name: &str, success: bool) {
        self.history.push((name.to_string(), success));
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    pub fn score(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            let successes = self.history.iter().filter(|(_, s)| *s).count();
            (successes as f64 / self.history.len() as f64) * 100.0
        }
    }

    pub fn history_summary(&self, n: usize) -> Vec<(String, bool)> {
        self.history.iter().rev().take(n).cloned().collect()
    }

    /// Check whether the produced output matches the expected answer.
    pub fn validate(&self, expected: &str, output: &str, tolerance: Option<f64>) -> bool {
        let tokens: Vec<&str> = output.trim().split_whitespace().collect();
        let expected = expected.trim();

        if let (Ok(b), Some(tol)) = (expected.parse::<f64>(), tolerance) {
            for token in &tokens {
                if let Ok(a) = token.parse::<f64>() {
                    if (a - b).abs() <= tol {
                        return true;
                    }
                }
            }
        } else {
            for token in &tokens {
                if token == &expected {
                    return true;
                }
            }
        }
        false
    }

    pub fn running_time_ms(&self) -> f64 {
        // Placeholder for future wall-clock tracking; returns a fixed default.
        1000.0
    }
}

/// A transfer-learning task: learn a skill from one training example and
/// generalize to a held-out test example in a new domain.
#[derive(Clone)]
pub struct TransferTask {
    pub domain: &'static str,
    pub description: &'static str,
    pub train_input: &'static str,
    pub train_output: &'static str,
    pub test_input: &'static str,
    pub test_output: &'static str,
}

/// Autonomous domain-transfer benchmark suite.
pub struct TransferSuite {
    pub tasks: Vec<TransferTask>,
    pub history: Vec<(String, bool)>,
    pub index: usize,
}

impl TransferSuite {
    pub fn new() -> Self {
        Self {
            tasks: vec![
                TransferTask {
                    domain: "vowel counting",
                    description: "Count the number of vowels in a lowercase English string.",
                    train_input: "hello",
                    train_output: "2",
                    test_input: "firefly agi",
                    test_output: "4",
                },
                TransferTask {
                    domain: "caesar cipher shift 3",
                    description: "Shift each letter forward by 3 in the alphabet (wrap around from z to a). Keep non-letters unchanged.",
                    train_input: "abc",
                    train_output: "def",
                    test_input: "xyz",
                    test_output: "abc",
                },
                TransferTask {
                    domain: "sum of digits",
                    description: "Sum all decimal digits in the input string, ignoring non-digit characters.",
                    train_input: "a1b2c3",
                    train_output: "6",
                    test_input: "fire5fly8agi2",
                    test_output: "15",
                },
                TransferTask {
                    domain: "count words",
                    description: "Count the number of whitespace-separated words in the input string.",
                    train_input: "hello world",
                    train_output: "2",
                    test_input: "firefly agi research runtime",
                    test_output: "4",
                },
                TransferTask {
                    domain: "title case",
                    description: "Convert the input string to title case: first letter of each word uppercase, the rest lowercase.",
                    train_input: "hello world",
                    train_output: "Hello World",
                    test_input: "firefly agi research",
                    test_output: "Firefly Agi Research",
                },
            ],
            history: Vec::new(),
            index: 0,
        }
    }

    pub fn next_task(&mut self) -> &TransferTask {
        let task = &self.tasks[self.index % self.tasks.len()];
        self.index += 1;
        task
    }

    pub fn record(&mut self, name: &str, success: bool) {
        self.history.push((name.to_string(), success));
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    pub fn score(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            let successes = self.history.iter().filter(|(_, s)| *s).count();
            (successes as f64 / self.history.len() as f64) * 100.0
        }
    }

    pub fn history_summary(&self, n: usize) -> Vec<(String, bool)> {
        self.history.iter().rev().take(n).cloned().collect()
    }
}
