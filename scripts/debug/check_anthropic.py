import os
import anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY")
print(f"Checking key: {api_key[:10]}...")

client = anthropic.Anthropic(api_key=api_key)

# Try a very basic model first
models_to_try = [
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1"
]

print("\nTesting Model Access:")
for model in models_to_try:
    try:
        print(f"  Trying {model}...", end="", flush=True)
        message = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f" ✅ SUCCESS (Output: {message.content[0].text})")
    except anthropic.NotFoundError:
        print(" ❌ NotFound (Not available to this key)")
    except anthropic.AuthenticationError:
        print(" ❌ AuthError (Key invalid)")
        break
    except Exception as e:
        print(f" ❌ Error: {e}")
