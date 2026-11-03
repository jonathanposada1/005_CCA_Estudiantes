#!/usr/bin/env python3
"""
Standalone sanity check for the Ollama connection used by this project.

It does two things:
  1. Hits Ollama's native /api/tags endpoint to confirm the host is reachable
     and lists which locally available models support tool calling.
  2. Uses aisuite (the same client the agents use) to send a real chat
     completion through the OpenAI-compatible /v1 endpoint, using LLM_MODEL.

Usage:
    python scripts/test_ollama_connection.py
    LLM_MODEL=ollama:qwen3:14b python scripts/test_ollama_connection.py

Exit code 0 on success, 1 on failure.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "openai:deepseek-r1:1.5b")


def check_ollama_reachable() -> bool:
    import requests

    tags_url = OLLAMA_API_URL.rstrip("/") + "/api/tags"
    print(f"1) Checking Ollama host at {tags_url} ...")
    try:
        resp = requests.get(tags_url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"   ❌ Could not reach Ollama: {e}")
        print("   - Is 'ollama serve' running on the host?")
        print("   - If running inside Docker, did you pass --add-host=host.docker.internal:host-gateway")
        print("     and set OLLAMA_API_URL=http://host.docker.internal:11434 ?")
        return False

    models = resp.json().get("models", [])
    if not models:
        print("   ⚠️  Ollama is reachable but no models are pulled yet.")
        print("       Run: ollama pull qwen3:4b-instruct")
        return False

    print(f"   ✅ Ollama reachable. {len(models)} model(s) available:")
    tool_capable = []
    for m in models:
        caps = m.get("capabilities", [])
        name = m.get("name", "?")
        marker = " (tools ✅)" if "tools" in caps else ""
        if "tools" in caps:
            tool_capable.append(name)
        print(f"      - {name}{marker}")

    if not tool_capable:
        print("   ⚠️  None of the available models advertise 'tools' capability.")
        print("       research_agent needs tool calling. Pull one, e.g.: ollama pull qwen3:4b-instruct")
    return True


def check_aisuite_chat_completion() -> bool:
    print(f"\n2) Sending a test chat completion via aisuite using model='{LLM_MODEL}' ...")
    try:
        from aisuite import Client
    except ImportError:
        print("   ❌ aisuite is not installed. Run: pip install -r requirements.txt")
        return False

    client = Client(
        provider_configs={
            "openai": {
                "api_key": "ollama",
                "base_url": OLLAMA_API_URL.rstrip("/") + "/v1",
                "timeout": 60,
            }
        }
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            temperature=0,
        )
        content = resp.choices[0].message.content
        print(f"   ✅ Model responded: {content!r}")
        return True
    except Exception as e:
        print(f"   ❌ aisuite call failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Ollama connection test ===")
    print(f"OLLAMA_API_URL={OLLAMA_API_URL}")
    print(f"LLM_MODEL={LLM_MODEL}\n")

    ok_reachable = check_ollama_reachable()
    ok_chat = check_aisuite_chat_completion() if ok_reachable else False

    print("\n=== Summary ===")
    print(f"Ollama reachable:        {'OK' if ok_reachable else 'FAIL'}")
    print(f"aisuite chat completion: {'OK' if ok_chat else 'FAIL'}")

    sys.exit(0 if (ok_reachable and ok_chat) else 1)
