import os
import anthropic

# Load API key from environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ ERROR: ANTHROPIC_API_KEY not set")
    print("Set it with: export ANTHROPIC_API_KEY='your-key'")
    print("Or create .secrets/anthropic_key file with your key")
    exit(1)

client = anthropic.Anthropic(api_key=api_key)

models_to_try = [
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

print(f"Testing key: {api_key[:20]}...")

for model in models_to_try:
    print(f"Trying {model}... ", end="", flush=True)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("✅ SUCCESS")
    except anthropic.NotFoundError:
        print("❌ Not Found (Model unavailable to this key)")
    except anthropic.AuthenticationError:
        print("❌ Auth Error (Key invalid)")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
