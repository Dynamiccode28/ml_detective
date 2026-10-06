"""
prompts.py

Prompt templates. Each one explicitly instructs the model to use ONLY
the provided facts and never introduce new numbers -- this is a
prompt-level safeguard, not a guarantee, which is why
report_builder.py also does a basic post-hoc number check.
"""

import json


def build_executive_summary_prompt(fact_packet: dict) -> str:
    return f"""You are a data quality analyst writing an executive summary.
Use ONLY the facts below. Do NOT invent any numbers, percentages, or statistics
not present in this data. Write 3-4 sentences, plain business language, no markdown headers.

FACTS:
{json.dumps(fact_packet, indent=2)}

Executive summary:"""


def build_section_narrative_prompt(section_title: str, items: list[dict]) -> str:
    if not items:
        return f"No {section_title.lower()} were found."
    return f"""You are a data quality analyst. Below is a list of findings with their
reasoning and recommended fixes. Rewrite them as a short, readable paragraph for
a {section_title} section of a report. Use ONLY the information given -- do not add
any numbers, percentages, or facts not present below. Keep it to 2-3 sentences per finding.

FINDINGS:
{json.dumps(items, indent=2)}

{section_title} section:"""


def build_next_steps_prompt(fact_packet: dict) -> str:
    return f"""Based on the health score and findings below, write a short, numbered
list (3-5 items) of concrete next steps a data scientist should take before training
a model on this dataset. Use ONLY the facts given, do not invent new issues.

FACTS:
{json.dumps(fact_packet, indent=2)}

Next steps:"""
