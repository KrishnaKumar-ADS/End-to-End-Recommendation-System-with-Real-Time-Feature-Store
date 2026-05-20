# What I Learned

## Delivery lessons

- A working feature without observability is difficult to trust.
- Production readiness requires explicit startup checks and failure modes.
- Test coverage grows fastest when each module has clear interfaces.

## Architecture lessons

- Hybrid retrieval (vector + lexical) is more stable than vector-only retrieval.
- Graph expansion improves contextual grounding for multi-hop questions.
- Separating routing, prompting, and retrieval simplifies troubleshooting.

## Operations lessons

- Health checks must use binaries available in the target image.
- Migration startup logic should handle multi-head and stale metadata states.
- Smoke tests should verify end-to-end flows, not only static health endpoints.

## Evaluation lessons

- Quality, latency, and cost must be measured together.
- TTFT is often more user-relevant than full completion latency.
- LLM-as-judge works best with strict JSON-only output constraints and retry logic.

## Next improvements

- Introduce reusable CI workflows to reduce duplication.
- Add stronger integration coverage for OpenAPI schema contracts.
- Add dashboard-style visualization for provider comparison trends over time.
- Expand smoke tests with a lightweight local mock mode for faster CI feedback.
