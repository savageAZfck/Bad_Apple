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
