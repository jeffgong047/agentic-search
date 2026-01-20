# Secrets Directory

This directory stores API keys and credentials locally. **Never commit this directory to git.**

## Setup

1. Copy the example files and add your keys:

```bash
cp anthropic_key.example anthropic_key
cp openai_key.example openai_key
cp postgres_password.example postgres_password
```

2. Edit each file and replace with your actual credentials.

3. Load the environment variables:

```bash
source load_env.sh
```

Or use the Python helper:

```python
from secrets_loader import load_secrets
load_secrets()
```

## Files

| File | Environment Variable | Description |
|------|---------------------|-------------|
| `anthropic_key` | `ANTHROPIC_API_KEY` | Anthropic/Claude API key |
| `openai_key` | `OPENAI_API_KEY` | OpenAI API key |
| `postgres_password` | `POSTGRES_PASSWORD` | PostgreSQL password |

## Security Notes

- This entire `.secrets/` directory is in `.gitignore`
- Never share these files or commit them
- Rotate keys if you suspect they've been exposed
- For production, use proper secret management (AWS Secrets Manager, Vault, etc.)
