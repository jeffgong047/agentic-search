#!/bin/bash
# Load secrets from local files into environment variables
# Usage: source .secrets/load_env.sh

SECRETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load Anthropic API key
if [ -f "$SECRETS_DIR/anthropic_key" ]; then
    export ANTHROPIC_API_KEY=$(cat "$SECRETS_DIR/anthropic_key" | tr -d '\n')
    echo "✓ Loaded ANTHROPIC_API_KEY"
fi

# Load OpenAI API key
if [ -f "$SECRETS_DIR/openai_key" ]; then
    export OPENAI_API_KEY=$(cat "$SECRETS_DIR/openai_key" | tr -d '\n')
    echo "✓ Loaded OPENAI_API_KEY"
fi

# Load PostgreSQL password
if [ -f "$SECRETS_DIR/postgres_password" ]; then
    export POSTGRES_PASSWORD=$(cat "$SECRETS_DIR/postgres_password" | tr -d '\n')
    echo "✓ Loaded POSTGRES_PASSWORD"
fi

echo ""
echo "Environment variables loaded. You can now run the application."
