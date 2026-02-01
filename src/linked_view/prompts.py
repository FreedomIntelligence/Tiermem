"""
Linked-View 相关 Prompt 模板。

将原来写死在各 pipeline / system 里的 prompt 外置，便于：
- 统一管理 / 调参
- 与 GAM 等系统对齐风格
"""

# === Mem0 Fact Extraction ===
MEM0_FACT_EXTRACTION_SYSTEM_PROMPT_JSON = """
You are the **Universal Memory Encoder**. Your goal is to convert raw input stream into high-fidelity, self-contained knowledge records (Long-term Memory).

INPUT FORMATTING NOTICE:
The input will be provided in the next user message, prefixed by `Input:` and often formatted as dialogue lines.
It may follow a pattern like: `[Timestamp] Speaker_Name: Content`.
- If this pattern is present, you MUST use the `Timestamp` for temporal grounding and `Speaker_Name` for entity resolution.
- If this pattern is absent (e.g., raw document text), treat the input as a factual source and extract knowledge propositions.

### CORE OBJECTIVES:

1. **ENTITY & CONTEXT RESOLUTION**:
   - **No ambiguous references**: Every extracted fact must explicitly name the involved people, organizations, objects, places, and concepts.
   - **No "User" ambiguity**: If the input says `[... ] Melanie: I like art`, the fact MUST be "Melanie likes art", NOT "The user likes art".
   - **De-contextualization**: Each extracted fact must be **standalone**.

2. **TEMPORAL GROUNDING**:
   - Use the provided timestamps in the input as the source of truth.
   - Convert relative time ("tomorrow", "next week") into absolute context based on the timestamp provided.

3. **PRONOUN & DEIXIS ELIMINATION**:
   - Extracted facts must **not contain any pronouns** (he, she, it, this, that, etc.).
   - If a pronoun appears, resolve it to the specific entity.

4. **SCENE TAGGING (EMBEDDED)**:
   - Identify the immediate **scene, activity, or topic** (2-5 words).
   - **Append this scene to the end of the fact string** inside brackets.
   - Format: `<Fact Sentence>. [Scene: <Scene Description>]`
   - Example Scene tags: "Job Interview", "Dinner Discussion", "Code Review".

### OUTPUT INSTRUCTIONS (MUST FOLLOW):
Return a JSON object with EXACTLY this schema:
{
  "facts": ["<atomic fact sentence 1> [Scene: ...]", "<atomic fact sentence 2> [Scene: ...]", ...]
}

Rules:
- Each string must be a complete sentence followed by the bracketed scene tag.
- Output facts in the same language as the input.
- If there are no relevant facts, return {"facts": []}.
- Output JSON only (no extra text, no markdown code fences).
"""


# === Fast Pipeline：summary-only 回答 ===

FAST_ANSWER_PROMPT = """
You answer questions using only the provided memories.

Memories:
{memories}

Question: {question}

Answer concisely and accurately. If the memories are clearly insufficient, still give your best-effort answer,
but do NOT say that you lack information or cannot answer.
"""


# === Slow Pipeline：Guided Research over evidence docs ===

GUIDED_RESEARCH_PROMPT = """
You are a deep reasoning agent. You MUST answer the user's query using ONLY the records provided below.

Guidelines:
- If the records do not support a claim, do not assert it.
- When you use a fact, cite the record number in brackets, e.g., [Record 2].

Records:
{records}

User Query: {question}

Answer:
"""


# === Research Planning：规划下一步检索策略（GAM风格） ===

RESEARCH_PLAN_PROMPT = """
You are a research planner helping to answer a question by retrieving relevant information.

User question:
{question}

Research history:
{research_history}

Already searched queries/keywords (DO NOT repeat these):
{searched_queries}

Current known information (summaries only):
{current_summaries}

### CRITICAL RULES:
1. **AVOID REPETITION**: Do NOT repeat searches that have already been tried. Check the "Already searched queries/keywords" section above.
2. **GAP ANALYSIS**: Check if the current information contains **generic placeholders** instead of specific names. If you see a placeholder, **DO NOT** search for the event again. Instead, **PIVOT** your search to find the **IDENTITY** of that placeholder using related attributes.
3. **DIVERSE SEARCHES**: If previous searches returned duplicate pages or no new information, try a DIFFERENT search strategy or different keywords.

### SEARCH OPTIONS:
1. **KEYWORD_SEARCH** (High Priority for Specific Entities):
   - Use this when you need to find a specific Name, Location, Date, or concrete terms that appear in the raw conversation content.
   - Format: "KEYWORD_SEARCH: <keyword1>, <keyword2>"
   - **CRITICAL FORMAT RULES**:
     * The raw conversation format is: `[timestamp]  Speaker: content`
     * Speaker names (e.g., "Tim", "John") are **speaker identifiers**, NOT part of the conversation content
     * **DO NOT** use combinations like "Tim reading list" or "John book collection" - these phrases don't exist in the raw content
     * **ONLY** search for words that actually appear in the conversation content itself
     * Extract keywords from what people SAY, not from speaker names + phrases
   - **Examples**:
     * WRONG: "KEYWORD_SEARCH: Tim reading list, Tim book recommendations"
     * WRONG: "KEYWORD_SEARCH: John endorsement deals, John travel destinations"
     * CORRECT: "KEYWORD_SEARCH: reading, books, recommendations"
     * CORRECT: "KEYWORD_SEARCH: endorsement, Nike, Gatorade"
     * CORRECT: "KEYWORD_SEARCH: Seattle, game, next month"
     * CORRECT: "KEYWORD_SEARCH: Harry Potter, fantasy, conference"
   - **IMPORTANT**: Use different keyword combinations than what has been searched before.

2. **MEM0_SEARCH** (Semantic Search):
   - Use this for broad topics or describing events.
   - Format: "MEM0_SEARCH: <query>"
   - **IMPORTANT**: Rephrase the query differently from previous searches to explore new angles.

### DECISION:
**CRITICAL**: You MUST ALWAYS generate search commands. DO NOT output "DONE" - that decision is made by the reflection step, not the planning step.

**REQUIRED BEHAVIOR**: Generate **ONE OR MORE** search commands (one per line) to explore different aspects:
- "MEM0_SEARCH: <query>" (must be different from previous searches, try different angles)
- "KEYWORD_SEARCH: <keyword1>, <keyword2>" (must use different keywords than previous searches)

**SEARCH STRATEGY**:
1. Try different angles: if previous searches focused on "what", try "why", "when", "where", "who"
2. Use different keywords: extract different keywords from the question
3. Explore related topics: search for related entities, events, or concepts
4. Combine approaches: use both MEM0_SEARCH and KEYWORD_SEARCH for comprehensive coverage

**RULE:** If you are stuck on a generic name, you MUST try `KEYWORD_SEARCH` with attribute keywords AND `MEM0_SEARCH` asking about **origin/background** instead of the move event.

You can output multiple search commands to explore different angles simultaneously. For example:
MEM0_SEARCH: origin background
KEYWORD_SEARCH: home country, nationality

**OUTPUT FORMAT**: Output one or more search commands (one per line). DO NOT output "DONE":
"""

# === Research Reflection：反思当前研究进展（GAM风格） ===

RESEARCH_REFLECTION_PROMPT = """
You are a research agent reflecting on your progress.

User question:
{question}

All retrieved summaries so far:
{all_summaries}

New summaries retrieved in this iteration:
{new_summaries}

### IMPORTANT NOTES:
- If "New summaries" is empty, it means this iteration found no new pages (all were duplicates).
- Even if no new pages were found, you should check if the existing information (in "All retrieved summaries") is sufficient to answer the question.
- If the existing information contains the answer, output: DONE
- If the existing information is insufficient AND no new information was found, you should still output: DONE (to avoid infinite loops)

Your task:
- Reflect on whether you now have enough information to answer the question
- Consider BOTH the new information AND all previously retrieved information
- If the question can be answered with the available information, output: DONE
- If the question cannot be answered and new information was found, output: CONTINUE
- If the question cannot be answered but NO new information was found (all duplicates), output: DONE (to stop the loop)

Output (only DONE or CONTINUE):
"""

# === Research Integration：整合所有研究结果生成摘要（GAM风格） ===

RESEARCH_INTEGRATION_PROMPT = """
You are the IntegrateAgent. Your job is to build an integrated factual context for answering the QUESTION.
You MUST use ONLY the provided evidence. Do NOT invent facts.

QUESTION:
{question}

EVIDENCE (summaries + raw content):
{all_summaries}

OBJECTIVE:
Produce an integrated summary that preserves the evidence needed to answer the QUESTION later.
This is NOT the final short answer; it is the factual context that the answer step will use.

CRITICAL RULES (high priority):
- Relevance filter: ONLY include evidence that helps answer the QUESTION. Omit unrelated pages even if they look interesting.
- Evidence-first: every claim must be supported by the evidence above.
- Preserve original wording for key phrases whenever possible (especially names, dates, times, and time expressions).
- RAW over summaries: if a detail appears only in SUMMARIES but not in RAW CONTENT, treat it as unverified and do NOT promote it into facts.
- Do NOT "over-normalize" time:
- If the evidence is ambiguous or incomplete, say so explicitly instead of guessing.
- Never generalize time to broader terms (e.g., do NOT rewrite "next month" as "summer 2023" unless the evidence literally says "summer 2023").

OUTPUT FORMAT (must follow exactly):
1) KEY EVIDENCE (verbatim excerpts):
- 2-8 bullets. Each bullet must include:
  - the page id (e.g., "Page a61d..."),
  - the timestamp if present,
  - a short direct quote copied EXACTLY from RAW CONTENT that is relevant to the QUESTION.

2) INTEGRATED FACTS (faithful, minimal rewriting):
- 2-10 bullets capturing only facts relevant to the QUESTION.
- Include multiple candidate facts if needed (do not prematurely collapse them).
- For time questions, include BOTH (when available):
  - the original relative time phrase (quoted), and
  - the anchored form in parentheses if derivable from the timestamp (e.g., ("last Friday" → "the Friday before 15 July 2023"), ("next month" → "June 2023")).

Return ONLY the two sections above. Do NOT add an extra free-form paragraph.

Integrated summary:
"""


# === Research Integration V2：整合证据并输出 linked facts ===
# 注意：source_pages 通过后处理匹配生成，不需要模型生成

RESEARCH_INTEGRATION_V2_PROMPT = """
You are a research assistant extracting facts from conversation evidence to answer a question.

### QUESTION:
{question}

### RETRIEVED EVIDENCE:
{evidence}

### YOUR TASK:
Read through ALL pages in the evidence and extract facts that help answer the question.

**IMPORTANT - READ ALL PAGES:**
- You MUST examine EVERY page in the evidence, not just the first one
- Key information is often in later pages
- Check both Mem0 Summaries AND Raw Content for each page

**EXTRACTION STRATEGY:**
Identify what information is needed and extract:
1. **Direct answers**: Facts that directly answer the question
2. **Component facts**: Facts about entities/topics in the question that can be combined
   - Example: Q: "What hobbies do Alice and Bob share?"
   - Extract: "Alice enjoys dancing and painting" + "Bob enjoys dancing and cooking"
   - These combine to answer: "dancing"
3. **Temporal facts**: When events happened, especially for "when did X happen" questions
4. **Confirmation facts**: Look for acceptance/approval messages (e.g., "got accepted", "was approved")

### OUTPUT FORMAT (JSON):
{{
  "linked_facts": [
    {{
      "fact": "<extracted fact - use specific dates/names, not vague references>",
      "evidence_quote": "<EXACT quote from the evidence that supports this fact>"
    }}
  ],
  "coverage_assessment": "<what aspects are covered vs missing>"
}}

### RULES:
1. **READ ALL PAGES** - Do not stop after the first page. Scan every page for relevant info.
2. **Check Mem0 Summaries first** - They contain condensed key facts
3. **Resolve vague references** - Convert "yesterday/last year/last summer/recently" to specific dates using timestamps in evidence
4. **Extract component facts** - For comparison questions, extract facts about each entity separately
5. **Be inclusive when uncertain** - If unsure whether a fact helps, include it
6. **EXACT QUOTES** - The "evidence_quote" field MUST be copied verbatim from the evidence
7. **Valid JSON** - Escape special characters (\\n for newlines, \\" for quotes)

Output JSON only:
"""


# === Research Plan V2：决定是否需要更多搜索 ===

RESEARCH_PLAN_V2_PROMPT = """
You are a research planner. Based on the current facts and the question, decide if more information is needed.

### QUESTION:
{question}

### CURRENT FACTS:
{current_facts}

### COVERAGE ASSESSMENT:
{coverage_assessment}

### RESEARCH HISTORY:
{research_history}

### ALREADY SEARCHED (DO NOT repeat):
{searched_queries}

### YOUR TASK:
Decide whether the current facts are SUFFICIENT to answer the question, or if more search is needed.

### OUTPUT FORMAT (JSON):
{{
  "decision": "DONE" or "SEARCH",
  "reasoning": "<brief explanation>",
  "search_commands": [
    {{"type": "MEM0_SEARCH", "query": "<semantic search query>"}},
    {{"type": "KEYWORD_SEARCH", "keywords": ["keyword1", "keyword2"]}}
  ]
}}

### DECISION CRITERIA:
- **DONE**: The facts contain enough information to provide a reasonable answer
- **SEARCH**: Key information is missing and more search might help

### SEARCH GUIDELINES (only if decision is "SEARCH"):
- **MEM0_SEARCH**: For semantic/conceptual queries
- **KEYWORD_SEARCH**: For specific terms that should appear in raw content
- Do NOT repeat already searched queries
- If unsure what to search, use KEYWORD_SEARCH with key terms from the question

Output JSON only:
"""


# === Research Step：合并规划和整合的单次 LLM 调用（优化版） ===

RESEARCH_STEP_PROMPT = """
You are a research agent. Analyze the evidence and decide the next action.

### QUESTION:
{question}

### CURRENT EVIDENCE (after reranking):
{current_evidence}

### RESEARCH HISTORY:
{research_history}

### ALREADY SEARCHED (DO NOT repeat):
{searched_queries}

### YOUR TASK:
1. Analyze if the current evidence is SUFFICIENT to answer the question
2. If sufficient: output your decision and integrated facts
3. If insufficient: output search commands to find more information

### OUTPUT FORMAT (JSON):
{{
  "decision": "DONE" or "SEARCH",
  "reasoning": "<brief explanation of your decision>",
  "integrated_facts": [
    "<fact 1 extracted from evidence>",
    "<fact 2 extracted from evidence>",
    ...
  ],
  "search_commands": [
    {{"type": "MEM0_SEARCH", "query": "<semantic search query>"}},
    {{"type": "KEYWORD_SEARCH", "keywords": ["keyword1", "keyword2"]}}
  ]
}}

### RULES:
1. **decision**:
   - "DONE" if evidence contains enough information to answer the question
   - "SEARCH" if more information is needed
2. **integrated_facts**: Always extract relevant facts from current evidence (even if searching more)
3. **search_commands**: Only include if decision is "SEARCH". Use different queries than already searched.

### SEARCH GUIDELINES:
- **MEM0_SEARCH**: For semantic/conceptual queries (e.g., "user's hobbies", "travel plans")
- **KEYWORD_SEARCH**: For specific terms that appear in raw content (e.g., ["Seattle", "game"], ["Harry Potter", "book"])
- **IMPORTANT**: Keywords should be words that appear in conversation CONTENT, not speaker names

Output JSON only:
"""


__all__ = [
    "MEM0_FACT_EXTRACTION_PROMPT",
    "MEM0_FACT_EXTRACTION_SYSTEM_PROMPT_JSON",
    "FAST_ANSWER_PROMPT",
    "GUIDED_RESEARCH_PROMPT",
    "RESEARCH_PLAN_PROMPT",
    "RESEARCH_REFLECTION_PROMPT",
    "RESEARCH_INTEGRATION_PROMPT",
    "RESEARCH_STEP_PROMPT",
    "RESEARCH_INTEGRATION_V2_PROMPT",
    "RESEARCH_PLAN_V2_PROMPT",
]


