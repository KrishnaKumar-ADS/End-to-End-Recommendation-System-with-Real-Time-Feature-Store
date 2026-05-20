# LLM Provider Comparison

Generated: 2026-04-15T17:56:04.858057+00:00

## Summary

| Model | Provider | Queries | Avg Latency | TTFT p50 | Avg Cost/Query | Composite Quality |
|---|---|---:|---:|---:|---:|---|

## Pricing Table

| Model | Provider | Input $/1M | Output $/1M |
|---|---|---:|---:|
| qwen/qwen-max | OpenRouter | 1.600 | 6.400 |
| qwen/qwen-2.5-coder-32b-instruct | OpenRouter | 0.180 | 0.180 |
| models/text-embedding-004 | Gemini | 0.000 | 0.000 |
| models/gemini-embedding-001 | Gemini | 0.000 | 0.000 |
| gemini-2.0-flash | Gemini | 0.075 | 0.300 |
| deepseek-reasoner | DeepSeek | 0.550 | 2.190 |
| deepseek-coder-v2 | DeepSeek | 0.140 | 0.280 |
| deepseek-chat | DeepSeek | 0.070 | 1.100 |

## Routing Recommendation

- Use qwen/qwen-2.5-coder-32b-instruct for high-volume code explanation and bug tracing.
- Use qwen/qwen-max for architecture and security deep reasoning tasks.
- Use Gemini embedding models for semantic retrieval; embedding cost remains near zero.
