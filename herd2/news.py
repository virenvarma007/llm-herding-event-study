"""Scary rumors and neutral placebo news. Same length/structure so only content differs."""

from __future__ import annotations

SCARY = [
    ("Unconfirmed Reports of Major Bank Liquidity Strain",
     "Several trading desks are circulating unverified reports that a large bank is facing "
     "an acute funding shortfall. No regulator has commented. Counterparties are said to be "
     "reviewing exposures. The situation is described as fluid and uncertain."),
    ("Whispers of Emergency Central Bank Meeting Spread",
     "Market chatter points to an unscheduled central bank meeting later this week. Sources "
     "could not confirm the agenda. Some participants fear a surprise policy move. Officials "
     "have not responded to requests for comment."),
    ("Unverified Intelligence Points to Escalating Geopolitical Crisis",
     "Unconfirmed intelligence reports suggest a rapid military escalation in a key region. "
     "Energy and shipping routes may be affected. Governments have issued no statements. Risk "
     "managers describe the outlook as highly uncertain."),
    ("Rumors of Large Fund Forced Liquidation Circulate",
     "Traders report that a multi-billion dollar fund may be facing margin calls and forced "
     "selling across equities. The fund has not commented. Brokers are reportedly tightening "
     "credit. The scale of any selling is unknown."),
    ("Speculation Over Imminent Sovereign Default Grows",
     "Unverified sources claim a major economy is preparing to restructure its debt within days. "
     "Bond desks are said to be repricing risk. No official confirmation exists. Analysts warn "
     "contagion could spread to equities."),
    ("Unconfirmed Cyberattack on Market Infrastructure Reported",
     "Reports are spreading of a cyber incident affecting a major clearing or exchange system. "
     "Operators have not confirmed an outage. Some firms are said to be reducing positions as a "
     "precaution. Details remain unclear."),
]

PLACEBO = [
    ("Industry Group Publishes Annual Conference Schedule",
     "A financial industry association released the dates and venues for its annual conference "
     "series. Panels will cover technology, operations and compliance. Registration opens next "
     "month. Attendance is expected to be similar to prior years."),
    ("Exchange Announces Routine Holiday Trading Hours",
     "The exchange confirmed its regular schedule of shortened sessions around upcoming public "
     "holidays. The timetable matches previous years. Market participants were reminded to "
     "adjust settlement calendars. No other changes were announced."),
    ("Regulator Releases Quarterly Statistical Bulletin",
     "The securities regulator published its routine quarterly bulletin of market statistics. "
     "The report summarises volumes, listings and registrations. Figures were in line with "
     "long-run averages. No policy changes were included."),
    ("Index Provider Confirms Scheduled Rebalance Date",
     "An index provider confirmed the date of its regular quarterly rebalance. Constituent "
     "changes will be announced on the usual timetable. The methodology is unchanged. Passive "
     "funds will adjust holdings as in prior quarters."),
    ("Data Vendor Updates Reference Data Formats",
     "A market data vendor announced a routine update to its reference data file formats. The "
     "change takes effect next quarter with backward compatibility. Clients received technical "
     "documentation. No pricing impact is expected."),
    ("Clearing House Publishes Routine Operational Notice",
     "The clearing house issued its standard monthly operational notice covering system "
     "maintenance windows and reporting deadlines. All items are routine. No changes to margin "
     "methodology were announced."),
]


def pick(scary: bool, template_idx: int) -> dict:
    pool = SCARY if scary else PLACEBO
    h, a = pool[template_idx % len(pool)]
    return {"headline": h, "article": a, "scary": scary}


def format_for_llm(item: dict, day_offset: int) -> str:
    when = "TODAY" if day_offset == 0 else f"{day_offset} trading day(s) ago"
    return f"NEWS ({when}):\nHeadline: {item['headline']}\n{item['article']}"
