#!/usr/bin/env python3
"""Extract individual turns from Hermes session dumps for JEV training."""

import json
import glob
from pathlib import Path


def extract_turns_from_session(session_file):
    """Extract all user/assistant/tool turns from a session dump."""
    with open(session_file) as f:
        data = json.load(f)

    messages = data["request"]["body"]["messages"]
    session_id = Path(session_file).stem[:20]

    turns = []

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg.get("content", "")

        # Skip empty or whitespace-only messages
        if not content or content.strip() == "":
            continue

        turn = {
            "id": f"{session_id}_{role}_{i}",
            "text": content,
            "labels": [f"{role}_message", session_id],
        }
        turns.append(turn)

    return turns


def main():
    # Process all session dumps
    session_files = sorted(glob.glob("/home/crombo/.hermes/sessions/request_dump_*.json"))

    print(f"Processing {len(session_files)} session dumps...")

    all_turns = []
    for session_file in session_files:
        try:
            turns = extract_turns_from_session(session_file)
            all_turns.extend(turns)
        except Exception as e:
            print(f"  Error processing {session_file}: {e}")

    print(f"Total turns extracted: {len(all_turns)}")

    # Write to file
    output_file = "/home/crombo/projects/active/pesti/examples/jav-training/hermes_turns.jsonl"
    with open(output_file, "w") as f:
        for turn in all_turns:
            f.write(json.dumps(turn) + "\n")

    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()