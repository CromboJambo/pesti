// Process Management Middleware for Hermes Agent
// Spawns, discovers, monitors, and attaches to running processes.
//
// Usage: hermes-process <command> [args...]
// Commands:
//   spawn <cmd>         - Generate command with PID capture wrapper
//   discover <pid>      - Check if process exists and get status
//   monitor <pid> <timeout_s> [interval_s] - Monitor with heartbeat feedback
//   attach <pid>        - Attach to running process for interaction

use std::process::{exit, Command};
use std::time::{Duration, Instant};

fn spawn_command(cmd: &str) -> String {
    format!("({}) & echo \"PID: $!\"", cmd)
}

fn discover_by_pid(pid: i32) -> Option<String> {
    let output = Command::new("ps")
        .args([
            "-p",
            &pid.to_string(),
            "-o",
            "pid,stat,etime,cmd",
            "--no-headers",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() || lines[0].trim().is_empty() {
        return None;
    }

    Some(lines[0].to_string())
}

fn monitor_process(pid: i32, timeout_s: u64, interval_s: u64) -> String {
    let start = Instant::now();

    loop {
        let elapsed = start.elapsed().as_secs();
        if elapsed >= timeout_s {
            return format!("Monitor result: timeout after {}s", elapsed);
        }

        match discover_by_pid(pid) {
            Some(_) => {
                println!("[{}s] Process still running (PID {})...", elapsed, pid);
                std::thread::sleep(Duration::from_secs(interval_s));
            }
            None => {
                return format!("Monitor result: exited after {}s", elapsed);
            }
        }
    }
}

fn attach_to_process(pid: i32) -> String {
    match discover_by_pid(pid) {
        Some(_) => {
            let msg = format!(
                "Process found. To debug interactively:\n  gdb -p {}\nOr to inspect memory:\n  cat /proc/{}/status",
                pid, pid
            );
            return msg;
        }
        None => {
            return format!("Process not found: {}", pid);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!("Usage: hermes-process <command> [args...]");
        println!("Commands: spawn|discover|monitor|attach");
        exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "spawn" => {
            // Join remaining args as the command to wrap
            let cmd = args[2..].join(" ");
            println!("{}", spawn_command(&cmd));
        }
        "discover" => {
            let pid: i32 = args[2].parse().expect("Invalid PID");
            match discover_by_pid(pid) {
                Some(info) => {
                    println!("Found process:");
                    println!("{}", info);
                }
                None => {
                    println!("Process not found or exited: {}", pid);
                    exit(1);
                }
            }
        }
        "monitor" => {
            let pid: i32 = args[2].parse().expect("Invalid PID");
            let timeout_s: u64 = if args.len() > 3 {
                args[3].parse().expect("Invalid timeout")
            } else {
                300
            };
            let interval_s: u64 = if args.len() > 4 {
                args[4].parse().expect("Invalid interval")
            } else {
                5
            };
            let result = monitor_process(pid, timeout_s, interval_s);
            println!("{}", result);
        }
        "attach" => {
            let pid: i32 = args[2].parse().expect("Invalid PID");
            let result = attach_to_process(pid);
            println!("{}", result);
        }
        _ => {
            println!("Unknown command. Use: spawn|discover|monitor|attach");
            exit(1);
        }
    }
}