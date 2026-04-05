"""Edge-case smoke tests against the auto-spawned llama-cpp-server."""
import json
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8100/v1/chat/completions"


def call(messages, max_tokens=10, temperature=0.0):
    req = urllib.request.Request(
        URL,
        data=json.dumps({
            "model": "m",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": {"status": e.code, "body": e.read().decode()[:200]}}


def main():
    print("\n=== Context length: ~2400 input tokens, 8192 ctx ===")
    long = "The word banana. " * 800
    r = call([{"role": "user", "content": long + " Count the bananas. Answer in one word."}])
    if "error" in r:
        print(f"  FAIL: {r['error']}")
    else:
        print(f"  prompt_tokens: {r['usage']['prompt_tokens']}")
        print(f"  response: {r['choices'][0]['message']['content']!r}")

    print("\n=== Context length: ~7500 input tokens (near 8192 ctx ceiling) ===")
    longer = "The word banana. " * 2500
    r = call([{"role": "user", "content": longer + " Answer in one word: apple."}])
    if "error" in r:
        print(f"  FAIL: {r['error']}")
    else:
        print(f"  prompt_tokens: {r['usage']['prompt_tokens']}")
        print(f"  response: {r['choices'][0]['message']['content']!r}")

    print("\n=== Context length: 10000+ tokens (should exceed 8192) ===")
    too_long = "The word banana. " * 3500
    r = call([{"role": "user", "content": too_long}])
    if "error" in r:
        print(f"  (expected error) {r['error']['status']}: {r['error']['body'][:150]}")
    else:
        print(f"  UNEXPECTED SUCCESS: {r['usage']}")

    print("\n=== Empty user message ===")
    r = call([{"role": "user", "content": ""}])
    if "error" in r:
        print(f"  error: {r['error']}")
    else:
        print(f"  response: {r['choices'][0]['message']['content']!r}")

    print("\n=== System prompt + user ===")
    r = call([
        {"role": "system", "content": "Respond only with the word GOOD."},
        {"role": "user", "content": "Hi"},
    ])
    if "error" in r:
        print(f"  error: {r['error']}")
    else:
        print(f"  response: {r['choices'][0]['message']['content']!r}")

    print("\n=== Stress: 10 sequential calls, mean latency ===")
    times = []
    for i in range(10):
        t0 = time.time()
        r = call([{"role": "user", "content": f"Say {i}."}], max_tokens=5)
        times.append(time.time() - t0)
    print(f"  mean: {sum(times)/len(times)*1000:.0f}ms, min: {min(times)*1000:.0f}ms, max: {max(times)*1000:.0f}ms")


if __name__ == "__main__":
    main()
