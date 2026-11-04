"""Teacher prompts. The verification prompt generalises IntelliSA's released Claude-4 teacher
prompt (replication/prompts/rq3_pseudo_labels/01_claude4_teacher_prompt.txt) to all nine smells;
the recovery prompt asks the same YES/NO question for a line no rule flagged."""
from __future__ import annotations

from resa.smells import DEFINITIONS, KEYWORDS, RULES

SYSTEM = "You are a static analyzer for Infrastructure-as-Code (IaC) security analysis."

VERIFY_TEMPLATE = """Task Description:
Your task is to analyze the raw code of an IaC script ({tech}) and decide whether a flagged line is a security smell. A static analysis tool has flagged a potential security issue, and your task is to determine if this is a true instance of the security smell or a false positive.

You must carefully examine the provided code snippet, consider its context and implementation intent, and make a binary decision: is this a genuine security smell or a false alarm?

---

Detection Rule: {smell}

Rule: {rule}

Keywords Heuristic: {keywords}

Definition: {definition}

---

Instructions:
1. Analyze line-by-line
2. Determine if the flagged code (the line marked with >>) matches the detection rule above
3. Consider the code's actual security implications, not just keyword presence
4. Account for context: Is this a placeholder, example, or documentation vs. actual configuration?
5. Focus on whether the flagged pattern poses a genuine security risk

---

Code Context: ({window}-line window around flagged line {line} of {file})

{context}

---

Output Constraint:
Answer with ONLY "YES" or "NO" - no explanations needed

YES = This is a true instance of {smell}
NO = This is a false positive"""

RECOVER_TEMPLATE = """Task Description:
Your task is to analyze the raw code of an IaC script ({tech}) and decide whether the marked line contains a security smell that a rule-based analyzer may have missed. The line was selected by a broad keyword pattern, so most selected lines are NOT smells.

---

Security Smell: {smell}

Definition: {definition}

Reference rule (GLITCH): {rule}

Keywords Heuristic: {keywords}

---

Instructions:
1. Focus on the line marked with >> (line {line} of {file})
2. Decide whether that line itself is a genuine instance of {smell} in active configuration or commands
3. Values coming from variables, lookups or templates are not hard-coded
4. Comments, examples and documentation are not configuration (except for the Suspicious comment smell)

---

Code Context: ({window}-line window around line {line})

{context}

---

Output Constraint:
Answer with ONLY "YES" or "NO" - no explanations needed

YES = The marked line is a true instance of {smell}
NO = It is not"""


def build_prompt(stage: str, tech: str, smell: str, file: str, line: int, context: str, window: int) -> str:
    template = VERIFY_TEMPLATE if stage == "verify" else RECOVER_TEMPLATE
    return template.format(
        tech=tech,
        smell=smell,
        rule=RULES[smell],
        keywords=KEYWORDS[smell],
        definition=DEFINITIONS[smell],
        window=2 * window + 1,
        file=file,
        line=line,
        context=context,
    )
