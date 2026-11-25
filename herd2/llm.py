"""LLM providers (Anthropic SDK, OpenRouter raw HTTP, deterministic mock), SQLite cache,
JSONL call log, and a shared spend ledger with hard caps."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .prompts import JSON_SCHEMA

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMResult:
    content: str
    parsed: Optional[dict]
    usage_in: int
    usage_out: int
    cost_usd: float
    latency_ms: float
    cached: bool
    model: str
    provider: str
    error: Optional[str] = None


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    try:
        i, j = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[i:j])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class FatalLLMError(RuntimeError):
    """Key/billing/permission problem: every further call would fail too, so stop the run instead of logging holds."""


class SpendLedger:
    """Append-only JSONL shared by all parallel runs; caps are checked against the sum."""

    def __init__(self, path: Path, caps: Dict[str, float]):
        self.path = path
        self.caps = {k: float(v) for k, v in caps.items()}
        self.lock = threading.Lock()
        self._pending: Dict[str, float] = {}
        self._last_check = 0.0
        self.totals: Dict[str, float] = {}

    def add(self, family: str, usd: float) -> None:
        with self.lock:
            self._pending[family] = self._pending.get(family, 0.0) + usd

    def flush(self, run: str) -> None:
        with self.lock:
            if not self._pending:
                return
            with open(self.path, "a") as f:
                for fam, usd in self._pending.items():
                    f.write(json.dumps({"t": time.time(), "run": run, "family": fam, "usd": usd}) + "\n")
            self._pending = {}

    def check(self) -> None:
        """Raise if any family exceeds its cap (re-reads the shared file)."""
        tot: Dict[str, float] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        tot[r["family"]] = tot.get(r["family"], 0.0) + float(r["usd"])
                    except Exception:
                        continue
        with self.lock:
            for fam, usd in self._pending.items():
                tot[fam] = tot.get(fam, 0.0) + usd
        self.totals = tot
        for fam, cap in self.caps.items():
            if tot.get(fam, 0.0) > cap:
                raise RuntimeError(f"SPEND CAP HIT: {fam} ${tot[fam]:.2f} > ${cap:.2f}. Stopping. Raise llm.spend_cap_usd in config.yaml to continue.")


class LLMClient:
    def __init__(self, cfg, run_dir: Path, run_name: str, provider_override: Optional[str] = None):
        self.cfg = cfg
        self.lcfg = cfg["llm"]
        self.run_name = run_name
        self.provider_override = provider_override      # e.g. "mock"
        self.max_tokens = int(self.lcfg["max_tokens"])
        self.log_path = run_dir / "llm_calls.jsonl"
        self.log_lock = threading.Lock()
        self.ledger = SpendLedger(Path(cfg["out_dir"]) / "spend_ledger.jsonl", self.lcfg["spend_cap_usd"])
        self.stats = {"calls": 0, "cached": 0, "parse_fail": 0, "errors": 0, "cost": 0.0, "in": 0, "out": 0}
        self.stats_lock = threading.Lock()
        self._anthropic = None
        self._http = None
        self._or_prices: Dict[str, Dict[str, float]] = {}
        self.cache = None
        if self.lcfg.get("cache", True):
            self.cache = sqlite3.connect(str(Path(cfg["out_dir"]) / "llm_cache.sqlite"), check_same_thread=False)
            self.cache.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, content TEXT, usage_in INT, usage_out INT, cost REAL, model TEXT)")
            self.cache.commit()
            self.cache_lock = threading.Lock()

    # ---- provider plumbing ----------------------------------------------
    def _anthropic_client(self):
        if self._anthropic is None:
            import anthropic  # lazy: not needed for mock runs
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set (export it or put it in v2_event_study/.env)")
            self._anthropic = anthropic.Anthropic(max_retries=10, timeout=120.0)   # SDK backs off on 429/5xx and honours retry-after
        return self._anthropic

    def _http_client(self):
        if self._http is None:
            import httpx
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY not set (export it or put it in v2_event_study/.env)")
            self._http = httpx.Client(timeout=120.0, headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "X-Title": "ai-herd-v2",
            })
        return self._http

    def provider_cfg(self, name: str) -> Dict[str, Any]:
        return dict(self.lcfg["providers"][name])

    @staticmethod
    def family(kind: str) -> str:
        return "anthropic" if kind == "anthropic" else ("openrouter" if kind == "openrouter" else "mock")

    # ---- cache ------------------------------------------------------------
    def _key(self, model: str, system: str, user: str) -> str:
        return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()

    def _cache_get(self, k: str):
        if self.cache is None:
            return None
        with self.cache_lock:
            row = self.cache.execute("SELECT content, usage_in, usage_out, cost, model FROM c WHERE k=?", (k,)).fetchone()
        return row

    def _cache_put(self, k: str, r: LLMResult):
        if self.cache is None or r.error:
            return
        with self.cache_lock:
            self.cache.execute("INSERT OR REPLACE INTO c VALUES (?,?,?,?,?,?)", (k, r.content, r.usage_in, r.usage_out, r.cost_usd, r.model))
            self.cache.commit()

    # ---- main entry -------------------------------------------------------
    def decide(self, provider: str, system: str, user: str, meta: Optional[dict] = None) -> LLMResult:
        pc = self.provider_cfg(provider)
        kind = self.provider_override or pc["kind"]
        model = "mock" if kind == "mock" else pc["model"]
        k = self._key(model, system, user)
        row = self._cache_get(k)
        if row is not None:
            content, ui, uo, cost, m = row
            r = LLMResult(content, _extract_json(content), ui, uo, 0.0, 0.0, True, m, provider)
        else:
            t0 = time.time()
            try:
                if kind == "anthropic":
                    r = self._call_anthropic(pc, system, user)
                elif kind == "openrouter":
                    r = self._call_openrouter(pc, system, user)
                elif kind == "mock":
                    r = self._call_mock(system, user)
                else:
                    raise ValueError(f"unknown provider kind {kind}")
            except FatalLLMError:
                raise
            except Exception as e:  # keep the simulation alive; count the error
                if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                    raise FatalLLMError(f"{type(e).__name__}: {e}") from e
                r = LLMResult("", None, 0, 0, 0.0, 0.0, False, model, provider, error=f"{type(e).__name__}: {e}")
            r.latency_ms = (time.time() - t0) * 1000
            r.provider = provider
            self._cache_put(k, r)
            if r.cost_usd:
                self.ledger.add(self.family(kind), r.cost_usd)
        with self.stats_lock:
            self.stats["calls"] += 1
            self.stats["cached"] += int(r.cached)
            self.stats["parse_fail"] += int(r.parsed is None and r.error is None)
            self.stats["errors"] += int(r.error is not None)
            self.stats["cost"] += r.cost_usd
            self.stats["in"] += r.usage_in
            self.stats["out"] += r.usage_out
        self._log(r, meta)
        return r

    def _log(self, r: LLMResult, meta: Optional[dict]):
        rec = {"t": time.time(), **(meta or {}), "provider": r.provider, "model": r.model, "cached": r.cached,
               "usage_in": r.usage_in, "usage_out": r.usage_out, "cost": r.cost_usd, "latency_ms": round(r.latency_ms, 1),
               "parsed": r.parsed, "error": r.error, "content": None if r.parsed else r.content[:2000]}
        with self.log_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # ---- providers --------------------------------------------------------
    _thinking_mode = "disabled"   # Haiku 5.5 accepts {"type":"disabled"} at effort<=high; no thinking tokens billed

    def _call_anthropic(self, pc: dict, system: str, user: str) -> LLMResult:
        import anthropic
        client = self._anthropic_client()
        # NOTE: structured-output json_schema adds ~365 input tokens per call (+56% cost); Haiku 5.5 returns
        # valid JSON from the prompt alone (measured), and the parser + circuit breaker catch any failure.
        kwargs = dict(
            model=pc["model"],
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if pc.get("json_schema_format"):
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": JSON_SCHEMA}}
        if self._thinking_mode == "disabled":
            kwargs["thinking"] = {"type": "disabled"}
        else:
            kwargs.setdefault("output_config", {})["effort"] = pc.get("effort", "low")
        resp = None
        for attempt in range(4):   # outer retry on top of the SDK's own: a long 429 storm should slow us, not kill the run
            try:
                resp = client.messages.create(**kwargs)
                break
            except anthropic.BadRequestError as e:
                # if this model rejects disabled thinking, fall back once to adaptive thinking at low effort
                if self._thinking_mode == "disabled" and "thinking" in str(e).lower():
                    self._thinking_mode = "adaptive"
                    return self._call_anthropic(pc, system, user)
                raise
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
                if attempt == 3:
                    raise
                time.sleep(20 * (attempt + 1))
        ui, uo = resp.usage.input_tokens, resp.usage.output_tokens
        cost = ui * pc["price_in_per_m"] / 1e6 + uo * pc["price_out_per_m"] / 1e6
        if resp.stop_reason == "refusal":
            return LLMResult("", None, ui, uo, cost, 0.0, False, pc["model"], "", error="refusal")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        parsed = _extract_json(text)
        err = None
        if resp.stop_reason == "max_tokens" and parsed is None:
            err = f"truncated at max_tokens={self.max_tokens} (raise llm.max_tokens)"
        return LLMResult(text, parsed, ui, uo, cost, 0.0, False, pc["model"], "", error=err)

    def _call_openrouter(self, pc: dict, system: str, user: str) -> LLMResult:
        http = self._http_client()
        body = {
            "model": pc["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": pc.get("temperature", 0.0),
            "response_format": {"type": "json_object"},
            "usage": {"include": True},
            # reasoning off by default: hidden reasoning tokens are billed as output and can blow the budget
            "reasoning": {"effort": pc["reasoning_effort"]} if pc.get("reasoning_effort") else {"enabled": False},
        }
        last_err = None
        for attempt in range(6):
            resp = http.post(OPENROUTER_URL, json=body)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(min(30, 2 ** attempt))
                continue
            if resp.status_code in (401, 402, 403):   # bad key, no credits, key spend limit hit: fatal, do not log as holds
                raise FatalLLMError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"OpenRouter error: {json.dumps(data['error'])[:300]}")
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
            u = data.get("usage", {}) or {}
            ui, uo = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
            cost = float(u.get("cost", 0.0) or 0.0)
            if cost == 0.0 and ui:
                pr = self._or_prices.get(pc["model"])
                cost = ui * pr["prompt"] + uo * pr["completion"] if pr else ui * 0.5e-6 + uo * 2.0e-6  # conservative guess
            parsed = _extract_json(text)
            err = None
            if parsed is None and choice.get("finish_reason") == "length":
                err = f"truncated at max_tokens={self.max_tokens} (model may be reasoning; raise llm.max_tokens or change model)"
            return LLMResult(text, parsed, ui, uo, cost, 0.0, False, pc["model"], "", error=err)
        raise RuntimeError(last_err or "OpenRouter: retries exhausted")

    def _call_mock(self, system: str, user: str) -> LLMResult:
        """Deterministic stand-in for pipeline tests. Sells more when scary news is present."""
        h = int(hashlib.md5(user.encode()).hexdigest(), 16)
        rng_val = (h % 1000) / 1000.0
        scary = "Unconfirmed" in user or "Rumors" in user or "Whispers" in user or "Speculation" in user
        trades = []
        tickers = [ln.split(" | ")[0] for ln in user.splitlines() if " | " in ln and not ln.startswith("ticker")]
        for i, t in enumerate(tickers):
            v = ((h >> (i * 7)) % 1000) / 1000.0
            if scary and v < 0.6:
                trades.append({"ticker": t, "action": "SELL", "fraction": 0.1})
            elif v < 0.15:
                trades.append({"ticker": t, "action": "BUY", "fraction": 0.05})
            elif v > 0.9:
                trades.append({"ticker": t, "action": "SELL", "fraction": 0.03})
        content = json.dumps({"trades": trades, "reason": "mock"})
        time.sleep(0.001 * (1 + rng_val))
        return LLMResult(content, json.loads(content), 450, 60, 0.0, 0.0, False, "mock", "")

    # ---- utilities --------------------------------------------------------
    def load_openrouter_prices(self) -> Dict[str, Dict[str, float]]:
        """Fetch per-token USD prices for the configured OpenRouter models (used only if usage.cost is absent)."""
        import httpx
        r = httpx.get("https://openrouter.ai/api/v1/models", timeout=60.0)
        r.raise_for_status()
        want = {p["model"] for p in self.lcfg["providers"].values() if p["kind"] == "openrouter"}
        out = {}
        for m in r.json().get("data", []):
            if m.get("id") in want:
                pr = m.get("pricing", {})
                out[m["id"]] = {"prompt": float(pr.get("prompt", 0)), "completion": float(pr.get("completion", 0)), "name": m.get("name")}
        self._or_prices = out
        return out

    def close(self):
        if self.cache is not None:
            self.cache.close()
        if self._http is not None:
            self._http.close()
