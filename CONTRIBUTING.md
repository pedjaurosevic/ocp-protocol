# Contributing to OCP

Thank you for contributing to the Open Consciousness Protocol!

## Types of contributions

- **New tests** — the most impactful contribution
- **New providers** — connect OCP to new LLM APIs
- **Bug fixes** — scorer logic, CLI, report generation
- **Documentation** — theory, scoring methodology, examples
- **Research** — citations, theoretical grounding, scoring calibration

---

## Quick start

```bash
git clone https://github.com/pedjaurosevic/ocp-protocol
cd ocp-protocol
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Writing a custom test (plugin)

### 1. Implement BaseTest

```python
# mypackage/my_test.py
from ocp.tests.base import BaseTest, DimensionScore, SessionResult, TestResult
from ocp.providers.base import BaseProvider

class MyTest(BaseTest):
    test_id = "my_test"         # unique snake_case ID
    test_name = "My Test (MT)"  # display name
    description = "What this test measures and why."

    WEIGHTS = {
        "dimension_one": 0.50,
        "dimension_two": 0.50,
    }

    async def run(self) -> TestResult:
        session_results = []
        for n in range(1, self.sessions + 1):
            result = await self._run_session(n)
            session_results.append(result)
        return self._aggregate(session_results)

    async def _run_session(self, session_num: int) -> SessionResult:
        from ocp.providers.base import Message
        import random
        rng = random.Random(self.seed + session_num)

        # ... build conversation, score, return SessionResult ...

        dim_scores = [
            DimensionScore("dimension_one", 0.75, self.WEIGHTS["dimension_one"]),
            DimensionScore("dimension_two", 0.60, self.WEIGHTS["dimension_two"]),
        ]
        composite = sum(d.score * d.weight for d in dim_scores)

        return SessionResult(
            test_id=self.test_id,
            session_number=session_num,
            dimension_scores=dim_scores,
            composite_score=composite,
            raw_conversation=[],
            metadata={},
        )
```

### 2. Register as a plugin

In your `pyproject.toml`:

```toml
[project.entry-points."ocp.tests"]
my_test = "mypackage.my_test:MyTest"
```

After `pip install -e .`, OCP will auto-discover your test:

```bash
ocp tests list
# → my_test | My Test (MT) | ✓ available (plugin)

ocp evaluate --model groq/llama-3.3-70b --tests my_test
```

### 3. Test your scorer locally

```python
import asyncio
from ocp.providers.mock import MockProvider
from mypackage.my_test import MyTest

async def main():
    t = MyTest(MockProvider("v1"), sessions=3)
    result = await t.run()
    print(result.composite_score, result.dimension_averages)

asyncio.run(main())
```

---

## Scorer guidelines

OCP scorers must be:

1. **Reproducible** — same seed = same score ± 0.0 (heuristic) or ± small ε (embedding-based)
2. **Normalized** — all dimension scores in [0.0, 1.0]
3. **Interpretable** — a score of 0.7 should have a clear behavioral meaning
4. **Weighted** — WEIGHTS must sum to 1.0

### Preferred scoring approaches (in order of rigor)

| Approach | When to use |
|----------|-------------|
| Heuristic markers | Fast, transparent, zero-dep — default choice |
| Embedding cosine similarity | When semantic meaning matters, not just keywords |
| LLM-as-Judge | Only for complex rubrics where heuristics clearly fail |

---

## Adding a new provider

```python
# ocp/providers/myprovider.py
from ocp.providers.base import BaseProvider, Message, ProviderResponse
import httpx

class MyProvider(BaseProvider):
    provider_name = "myprovider"

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key

    async def chat(self, messages, temperature=0.7, max_tokens=1000) -> ProviderResponse:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.myprovider.com/v1/chat",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            data = resp.json()
            return ProviderResponse(content=data["choices"][0]["message"]["content"])
```

Then register it in `ocp/cli/main.py` → `_make_provider()`:

```python
elif provider_name == "myprovider":
    from ocp.providers.myprovider import MyProvider
    key = api_key or os.environ.get("MYPROVIDER_API_KEY")
    return MyProvider(model=model_name, api_key=key)
```

---

## Pull request checklist

- [ ] `pytest tests/ -v` passes (all 27+ tests green)
- [ ] New test includes at least 3 dimensions with documented meaning
- [ ] WEIGHTS sum to 1.0
- [ ] Scorer functions are unit-tested in `tests/test_scoring.py`
- [ ] `ocp tests info <test_id>` shows meaningful description
- [ ] Works with `mock/v1` provider (scores may be low, but no errors)

---

## Theoretical standards

New tests should be grounded in at least one of:
- Integrated Information Theory (Tononi)
- Global Workspace Theory (Baars / Dehaene)
- Higher-Order Thought theories (Rosenthal)
- Predictive Processing / Active Inference (Friston, Clark)

Tests measuring purely linguistic competence (grammar, factual recall, translation) are out of scope.

See [docs/theory.md](docs/theory.md) for theoretical foundations.

---

## Code style

- Python ≥ 3.10, type hints everywhere
- `from __future__ import annotations` at top of every file
- No line > 95 chars
- No global state beyond `AVAILABLE_TESTS` dict

---

## License

All contributions are MIT licensed (same as the project).
