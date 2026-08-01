#!/usr/bin/env python3
"""Benchmark llama-server generation speed + SSD reads per run.

Measures disk reads via /proc/<pid>/io read_bytes deltas around each
400-token generation. On a fully RAM-resident model the delta decays
toward ~0 as cold MoE experts fault into page cache.

Usage: python3 bench.py [--url http://127.0.0.1:58190] [--runs 4] [--max-tokens 400]
"""
import argparse, json, subprocess, urllib.request

PROMPTS = [
    "Write a 150-word short story about a robot learning to paint.",
    "Explain quantum entanglement to a 10-year-old in a few paragraphs.",
    "Write a 150-word short story about a robot learning to paint.",  # repeat: warm-cache check
    "Describe the history of the transistor in three paragraphs.",
]

def server_pid():
    out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("llama-server not running")
    return int(out.stdout.split()[0])

def read_bytes(pid):
    with open(f"/proc/{pid}/io") as f:
        for line in f:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return 0

def rss_gib(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1048576

def chat(url, prompt, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:58190")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    pid = server_pid()
    print(f"llama-server pid {pid}, RSS {rss_gib(pid):.1f} GiB")
    print(f"{'run':>3} {'tok/s':>7} {'prompt t/s':>10} {'SSD read GiB':>12} {'RSS GiB':>8}")
    for i in range(args.runs):
        r1 = read_bytes(pid)
        d = chat(args.url, PROMPTS[i % len(PROMPTS)], args.max_tokens)
        r2 = read_bytes(pid)
        t = d.get("timings", {})
        print(f"{i+1:>3} {t.get('predicted_per_second', 0):>7.2f} "
              f"{t.get('prompt_per_second', 0):>10.2f} "
              f"{(r2-r1)/2**30:>12.2f} {rss_gib(pid):>8.1f}")

if __name__ == "__main__":
    main()
