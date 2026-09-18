#!/usr/bin/env python3
"""Talk to any model on OpenRouter from the terminal. Standard library only.

Setup (once):
    export OPENROUTER_API_KEY="sk-or-v1-..."
  ...or put OPENROUTER_API_KEY=sk-or-v1-... in a .env file next to this script.

Use:
    python chat_openrouter.py                                  # interactive chat
    python chat_openrouter.py -m anthropic/claude-sonnet-4.5   # pick the model
    python chat_openrouter.py "what does the NYC Charter say?"  # one-shot answer
    cat notes.txt | python chat_openrouter.py "summarize this"  # pipe in context
    python chat_openrouter.py --list qwen                       # find model ids

In-chat commands:
    /model <id>     switch models mid-conversation (history carries over)
    /models [text]  search the live model catalogue
    /system <text>  set or replace the system prompt
    /clear          forget the conversation, keep the system prompt
    /save <file>    write the transcript to a file
    /exit           quit (Ctrl-D or Ctrl-C also work)

Import it, too:
    from chat_openrouter import ask
    ask("hello", model="google/gemini-2.5-flash")
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.5")


# ---------------------------------------------------------------- credentials

def load_key(cli_key=None):
    """Find the API key: --key, then $OPENROUTER_API_KEY, then a .env file."""
    if cli_key:
        return _check(cli_key)
    if os.getenv("OPENROUTER_API_KEY"):
        return _check(os.environ["OPENROUTER_API_KEY"])

    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), ".env", os.path.expanduser("~/.openrouter_key")):
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            text = fh.read()
        if path.endswith(".env"):
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return _check(line.split("=", 1)[1].strip().strip("'\""))
        elif text.strip():
            return _check(text.strip())

    sys.exit(
        "No API key found. Run:\n"
        '    export OPENROUTER_API_KEY="sk-or-v1-..."\n'
        "or pass --key sk-or-v1-..., or put it in a .env file next to this script."
    )


def _check(key):
    """Catch the placeholder before the API answers with a confusing 401."""
    if "REPLACE_ME" in key or not key.startswith("sk-or-"):
        sys.exit(
            f"That is not a real API key ({key[:12]}...).\n"
            "Open the .env file next to this script and replace the placeholder "
            "with your key from https://openrouter.ai/keys"
        )
    return key


# ------------------------------------------------------------------- HTTP I/O

def _open(path, key, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        API_BASE + path,
        data=data,
        method="POST" if data is not None else "GET",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "chat_openrouter.py",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _explain(err):
    """Turn an HTTPError into the message OpenRouter actually sent back."""
    try:
        body = json.loads(err.read().decode())
        return body.get("error", {}).get("message") or json.dumps(body)
    except Exception:
        return str(err)


# ------------------------------------------------------------------- the chat

def complete(messages, model=DEFAULT_MODEL, key=None, temperature=None,
             max_tokens=None, stream=True, out=sys.stdout):
    """Send `messages` to `model` and return the reply text.

    With stream=True the reply is printed to `out` as it arrives.
    """
    key = load_key(key)
    payload = {"model": model, "messages": messages, "stream": bool(stream)}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens:
        payload["max_tokens"] = max_tokens

    try:
        response = _open("/chat/completions", key, payload)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"OpenRouter error {err.code}: {_explain(err)}") from None
    except urllib.error.URLError as err:
        raise RuntimeError(f"Could not reach OpenRouter: {err.reason}") from None

    if not stream:
        body = json.loads(response.read().decode())
        return body["choices"][0]["message"]["content"]

    pieces = []
    for raw in response:                       # server-sent events, one per line
        line = raw.decode("utf-8").strip()
        if not line or line.startswith(":"):   # keep-alive comment
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        piece = (choices[0].get("delta") or {}).get("content")
        if piece:
            pieces.append(piece)
            out.write(piece)
            out.flush()
    out.write("\n")
    out.flush()
    return "".join(pieces)


def ask(prompt, model=DEFAULT_MODEL, system=None, **kwargs):
    """One-shot convenience wrapper: ask('hi', model='openai/gpt-4o')."""
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    kwargs.setdefault("stream", False)
    return complete(messages, model=model, **kwargs)


def list_models(key=None, query=""):
    """Print model ids (optionally filtered) with context size and input price."""
    key = load_key(key)
    try:
        catalogue = json.loads(_open("/models", key).read().decode())["data"]
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"OpenRouter error {err.code}: {_explain(err)}") from None

    needle = query.lower()
    rows = [m for m in catalogue
            if needle in m["id"].lower() or needle in m.get("name", "").lower()]
    for model in sorted(rows, key=lambda m: m["id"]):
        context = model.get("context_length") or 0
        price = float(model.get("pricing", {}).get("prompt") or 0) * 1_000_000
        print(f"{model['id']:<55} {context:>9,} ctx  ${price:>7.2f}/M in")
    print(f"\n{len(rows)} of {len(catalogue)} models"
          + (f" matching {query!r}" if query else ""))


# ------------------------------------------------------------- interactive UI

def repl(model, system, key, temperature, max_tokens, stream):
    messages = [{"role": "system", "content": system}] if system else []
    print(f"OpenRouter chat — model: {model}")
    print("Commands: /model <id>  /models [text]  /system <text>  /clear  /save <file>  /exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue

        command, _, argument = user_input.partition(" ")
        argument = argument.strip()

        if command in ("/exit", "/quit", "exit", "quit"):
            return
        if command == "/model":
            if argument:
                model = argument
                print(f"[model: {model}]\n")
            else:
                print(f"[model: {model}]\n")
            continue
        if command == "/models":
            try:
                list_models(key, argument)
            except RuntimeError as err:
                print(err)
            print()
            continue
        if command == "/system":
            messages = [m for m in messages if m["role"] != "system"]
            if argument:
                messages.insert(0, {"role": "system", "content": argument})
            print(f"[system prompt {'set' if argument else 'cleared'}]\n")
            continue
        if command == "/clear":
            messages = [m for m in messages if m["role"] == "system"]
            print("[history cleared]\n")
            continue
        if command == "/save":
            path = argument or "transcript.md"
            with open(path, "w") as fh:
                for message in messages:
                    fh.write(f"### {message['role']}\n\n{message['content']}\n\n")
            print(f"[saved {len(messages)} messages to {path}]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        print("Assistant: ", end="", flush=True)
        try:
            reply = complete(messages, model=model, key=key, temperature=temperature,
                             max_tokens=max_tokens, stream=stream)
        except RuntimeError as err:
            print(f"\n{err}\n")
            messages.pop()                     # don't poison the history
            continue
        except KeyboardInterrupt:
            print("\n[interrupted]\n")
            messages.pop()
            continue
        if not stream:
            print(reply)
        print()
        messages.append({"role": "assistant", "content": reply})


def main():
    parser = argparse.ArgumentParser(
        description="Chat with any OpenRouter model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model ids look like: anthropic/claude-sonnet-4.5, openai/gpt-4o,\n"
               "google/gemini-2.5-flash, qwen/qwen3-235b-a22b, meta-llama/llama-3.3-70b-instruct.\n"
               "Run --list to see what your key can reach.",
    )
    parser.add_argument("prompt", nargs="*", help="one-shot prompt (omit for interactive chat)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"model id (default: {DEFAULT_MODEL})")
    parser.add_argument("-s", "--system", help="system prompt")
    parser.add_argument("-t", "--temperature", type=float, help="sampling temperature")
    parser.add_argument("--max-tokens", type=int, help="cap the reply length")
    parser.add_argument("--key", help="API key (default: $OPENROUTER_API_KEY or .env)")
    parser.add_argument("--no-stream", action="store_true", help="wait for the full reply")
    parser.add_argument("--list", nargs="?", const="", metavar="TEXT",
                        help="list available models, optionally filtered by TEXT")
    args = parser.parse_args()

    key = load_key(args.key)

    if args.list is not None:
        list_models(key, args.list)
        return

    prompt = " ".join(args.prompt)
    if not sys.stdin.isatty():                 # piped input becomes context
        piped = sys.stdin.read().strip()
        if piped:
            prompt = f"{prompt}\n\n{piped}".strip()

    if prompt:
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages.append({"role": "user", "content": prompt})
        try:
            reply = complete(messages, model=args.model, key=key, temperature=args.temperature,
                             max_tokens=args.max_tokens, stream=not args.no_stream)
        except RuntimeError as err:
            sys.exit(str(err))
        if args.no_stream:
            print(reply)
        return

    repl(args.model, args.system, key, args.temperature, args.max_tokens, not args.no_stream)


if __name__ == "__main__":
    main()
