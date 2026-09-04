//! Bad Apple MCP server binary.
//!
//! `badapple-mcp stdio` runs the Model Context Protocol over stdin/stdout.
//! `badapple-mcp socket` listens on a Unix domain socket (default `/var/run/badapple/mcp.sock`).
//!
//! This server is off by default for air-gap certification.

use anyhow::Result;
use bad_apple::mcp::McpServer;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("stdio");

    let socket_path = std::env::var("BADAPPLE_MCP_SOCKET_PATH")
        .unwrap_or_else(|_| "/var/run/badapple/mcp.sock".to_string());

    let server = McpServer::new(socket_path);

    match mode {
        "stdio" => server.serve_stdio(),
        "socket" => server.serve_unix(),
        _ => {
            eprintln!("usage: badapple-mcp [stdio|socket]");
            std::process::exit(1);
        }
    }
}
