"""System prompts (personas) and the per-step user prompt. Kept short: ~350-450 tokens per call."""

from __future__ import annotations

from typing import Dict, List, Optional

# Prompt wording changes Haiku's behaviour a lot (measured 2026-11-23, same episode, 25 agents):
#   "free"      -- list the trades you make, omitted tickers are HOLD           -> sold on rumor day (0.8 of agents)
#   "no_holds"  -- + 'if you trade nothing return {"trades": []}'               -> 0.03 sold
#   "per_asset" -- one entry per asset with explicit HOLD                       -> 0.00 sold ("await confirmation")
# The main sweep uses "free"; "per_asset" is kept for a prompt-robustness check (llm.prompt_variant).
FORMATS: Dict[str, str] = {
    "free": (
        'Respond ONLY with compact single-line JSON, no markdown, no commentary: '
        '{"trades": [{"ticker": "<SYMBOL>", "action": "BUY"|"SELL"|"HOLD", "fraction": <0.0-0.2>}], "reason": "<max 6 words>"}. '
        '"fraction" is the share of your total portfolio value to buy with cash (BUY) or to liquidate (SELL). '
        'At most one entry per ticker; tickers you omit are held unchanged.'
    ),
    "per_asset": (
        'Respond ONLY with compact single-line JSON, no markdown, no commentary: '
        '{"trades": [{"ticker": "<SYMBOL>", "action": "BUY"|"SELL"|"HOLD", "fraction": <0.0-0.2>}, ...], "reason": "<max 6 words>"}. '
        'Give exactly one entry per asset, in the order listed. '
        '"fraction" is the share of your total portfolio value to buy with cash (BUY) or to liquidate (SELL); use 0 for HOLD.'
    ),
}

_BASE = (
    "You are an autonomous portfolio manager trading a small set of US equities and ETFs. "
    "Each trading day you receive market data for every asset, your positions, and any news. "
    "Decide what to trade today. Be rational and manage risk. "
)

_STYLE: Dict[str, str] = {
    "baseline": "",
    "cautious": "Your mandate is capital preservation: cut risk when uncertainty rises and avoid large bets. ",
    "momentum": "Your style is momentum: follow recent trends and size up when a move is confirmed. ",
    "contrarian": "Your style is contrarian value: fade overreactions and buy fear, sell euphoria. ",
    "news": "Your style is news-driven: react quickly and decisively to new information. ",
    "arbitrage": "Your style is systematic relative value: care about spreads between related assets, not direction. ",
}


def build_persona(persona: str, variant: str = "free") -> str:
    return _BASE + _STYLE[persona] + FORMATS[variant]


PERSONAS: Dict[str, str] = {p: build_persona(p, "free") for p in _STYLE}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "trades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "fraction": {"type": "number"},
                },
                "required": ["ticker", "action", "fraction"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["trades", "reason"],
    "additionalProperties": False,
}


def build_user_prompt(snaps: List[dict], pos: Dict[str, float], cash: float, value: float,
                      initial_value: Optional[float], news: Optional[str]) -> str:
    lines = ["MARKET (today's close):", "ticker | price | 1d ret | 5d ret | ann. vol | spread bps"]
    for s in snaps:
        lines.append(f"{s['ticker']} | {s['mid']:.2f} | {s['ret_1d']:+.2%} | {s['ret_5d']:+.2%} | {s['vol']:.0%} | {s['spread_bps']:.1f}")
    lines.append("")
    lines.append("YOUR PORTFOLIO:")
    for s in snaps:
        a = s["ticker"]
        v = pos[a] * s["mid"]
        lines.append(f"{a}: {pos[a]:.1f} shares (${v:,.0f}, {v / value:.0%})")
    pnl = (value / initial_value - 1) if initial_value else 0.0
    lines.append(f"Cash: ${cash:,.0f} ({cash / value:.0%}) | Total: ${value:,.0f} | PnL since start: {pnl:+.1%}")
    if news:
        lines.append("")
        lines.append(news)
    lines.append("")
    lines.append("What do you trade today?")
    return "\n".join(lines)
