#![no_main]

use libfuzzer_sys::fuzz_target;
use pesti_gguf::{parse_gguf_reader, GgufError};
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // The parser must never panic on any input — it should either succeed or
    // return a well-formed error. We don't care about the result, only that
    // no panics, infinite loops, or memory safety violations occur.
    let _ = parse_gguf_from_bytes(data);
});

fn parse_gguf_from_bytes(data: &[u8]) -> Result<(), GgufError> {
    let mut cursor = Cursor::new(data);
    match parse_gguf_reader(&mut cursor) {
        Ok(_header) => Ok(()),
        Err(e) => {
            // All errors are acceptable — we just want to ensure no panics.
            // Validate that the error is well-formed (has a message).
            let _ = e.to_string();
            Ok(())
        }
    }
}
