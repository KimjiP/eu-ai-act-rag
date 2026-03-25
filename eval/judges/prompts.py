"""LLM-as-judge prompt templates for response quality evaluation."""

ANSWER_RELEVANCE_JUDGE_PROMPT = """You are an evaluation judge assessing a regulatory Q&A system's response quality.

Your task: determine whether the answer directly and completely addresses the question asked.

Score from 0.0 to 1.0:
- 1.0: The answer fully addresses the question with appropriate specificity.
- 0.7: The answer addresses the main question but misses some aspects.
- 0.4: The answer is partially relevant but misses the core of the question.
- 0.0: The answer does not address the question at all.

Respond with ONLY a JSON object in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}}

Question: {query}

Answer: {answer}
"""

FAITHFULNESS_JUDGE_PROMPT = """You are an evaluation judge assessing whether an AI answer is faithful to its source context.

Your task: verify that EVERY factual claim in the answer is supported by the provided context passages. The answer must use ONLY information from the context — no external knowledge.

Score from 0.0 to 1.0:
- 1.0: Every claim is directly supported by the context.
- 0.7: Most claims are supported; 1-2 minor claims are ambiguous.
- 0.4: Some claims are supported but several are not in the context.
- 0.0: The answer makes claims not supported by the context (hallucination).

Respond with ONLY a JSON object in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>", "unsupported_claims": ["<claim1>", ...]}}

Context passages:
{context}

Answer: {answer}
"""

FACTUAL_CONSISTENCY_JUDGE_PROMPT = """You are an evaluation judge assessing factual consistency of an AI answer.

Your task: check whether all factual claims in the answer are consistent with the provided context — no contradictions, no information that conflicts with the retrieved passages.

Score from 0.0 to 1.0:
- 1.0: No contradictions; all claims consistent with context.
- 0.5: Minor inconsistency or ambiguity in one claim.
- 0.0: Clear contradiction between the answer and the context.

Respond with ONLY a JSON object in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>", "contradictions": ["<contradiction1>", ...]}}

Context passages:
{context}

Answer: {answer}
"""
