# Compresr SDK


[![PyPI](https://img.shields.io/pypi/v/compresr)](https://pypi.org/project/compresr/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://pypi.org/project/compresr/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Official SDKs for [Compresr](https://compresr.ai) - Intelligent context compression to reduce LLM API costs by 30-70%.

## SDKs

| Language | Package | Documentation |
|----------|---------|---------------|
| Python | [![PyPI](https://img.shields.io/pypi/v/compresr)](https://pypi.org/project/compresr/) | [python/README.md](python/README.md) |
| curl/REST | - | [curl/README.md](curl/README.md) |

## Quick Start

### Python

```bash
pip install compresr
```

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

result = client.compress(
    context="Your long context text...",
    compression_model_name="espresso_v1"
)

print(f"Saved {result.data.tokens_saved} tokens!")
```

### curl

```bash
curl -X POST https://api.compresr.ai/api/compress/question-agnostic/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Your long context text...",
    "compression_model_name": "espresso_v1"
  }'
```

## Getting Your API Key

1. Create an account at [compresr.ai](https://compresr.ai)
2. Navigate to Dashboard → API Keys
3. Click "Create New Key" and copy it (shown only once!)

## Features

- **Token Savings**: Reduce context size by 30-70% — cut LLM API costs on every call
- **Two modes**: **CompressionClient** rewrites text at the token level (system prompts, documents, retrieved passages); **FilterClient** keeps or drops entire chunks from a retrieval set
- **Query-aware**: Optionally pass a query so compression preserves answer-relevant information
- **Streaming**: Real-time streaming for both compression and filtering
- **Async**: Full async/await support for high-throughput pipelines

## Documentation

- [API Documentation](https://compresr.ai/docs)
- [Python SDK](python/README.md)
- [curl/REST Examples](curl/README.md)

### Run CI Locally

Test all workflows locally before pushing:

```bash
# Auto-installs act and runs all CI workflows
./test_ci_local.sh
```

On first run, it will install [act](https://github.com/nektos/act)

For tests requiring API key, create `.secrets` file:
```bash
echo "COMPRESR_API_KEY=your_key" > .secrets
```

**Note:** Requires [GitHub CLI](https://cli.github.com/)

## Support

Need help or have questions?

- Email: support@compresr.ai
- [API Documentation](https://compresr.ai/docs)
- [GitHub Discussions](https://github.com/compresr/sdk/discussions)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
