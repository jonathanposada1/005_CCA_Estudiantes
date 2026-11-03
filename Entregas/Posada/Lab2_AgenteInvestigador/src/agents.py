import os
from datetime import datetime
from urllib import response
from aisuite import Client
from src.research_tools import (
    arxiv_search_tool,
    tavily_search_tool,
    wikipedia_search_tool,
)
from src.optica_tools import (
    optica_open_access_search,
    site_search,
    check_open_access,
)

# === LLM provider config ===
# NOTE: the installed aisuite version's native "ollama:" provider calls Ollama's
# native /api/chat endpoint and forwards OpenAI-style kwargs (temperature,
# max_tokens, tool_choice, ...) as-is, which Ollama rejects with 400 Bad Request.
# Workaround: route through aisuite's "openai" provider (the real OpenAI Python
# SDK, which supports these kwargs properly) but pointed at Ollama's own
# OpenAI-compatible /v1 endpoint instead of api.openai.com. This means the
# "openai" provider slot is now dedicated to local Ollama; to reconnect to real
# GPT models later, remove/replace the provider_configs below and set
# OPENAI_API_KEY instead.
# Model string format for aisuite: "<provider>:<model-name>"
# Override the model with env var LLM_MODEL, e.g. "openai:qwen3:14b".
# Default uses a custom Ollama model tag (qwen3-4b-instruct-16k) created via a
# Modelfile with a larger num_ctx (16384) than the base qwen3:4b-instruct's
# default 4096 tokens -- the agent workflow's accumulated context (research +
# draft + history) regularly exceeds 4096 and errors with
# "exceed_context_size_error". See README.md "Increasing Ollama's context size".
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai:qwen3-4b-instruct-16k")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

# Local models can take much longer than cloud APIs to finish a long
# generation (e.g. the writer_agent's full report). Override with
# OLLAMA_TIMEOUT (seconds).
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Max tokens the writer may generate. Must leave room for the prompt inside
# the model's context window (16k for the *-16k models).
WRITER_MAX_TOKENS = int(os.getenv("WRITER_MAX_TOKENS", "4096"))

# Qwen3 hybrid models (e.g. qwen3:1.7b) "think" before answering, which wastes
# time/tokens and breaks the planner's list parsing. "/no_think" is Qwen3's
# switch to skip it; set NO_THINK=0 to keep thinking on.
NO_THINK = os.getenv("NO_THINK", "1") == "1"


def no_think(text: str) -> str:
    return f"{text}\n\n/no_think" if NO_THINK else text


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks if the model still emits them."""
    import re

    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


client = Client(
    provider_configs={
        "openai": {
            "api_key": "ollama",  # Ollama ignores it, but the OpenAI SDK requires a value
            "base_url": OLLAMA_API_URL.rstrip("/") + "/v1",
            "timeout": OLLAMA_TIMEOUT,
            "max_retries": 0,  # a local timeout won't fix itself; fail fast instead of 3x the wait
        }
    }
)


# Streaming client for long generations (writer/editor). With stream=True the
# timeout applies between chunks, not to the whole answer, so a slow local model
# can take as long as it needs as long as it keeps producing text.
from openai import OpenAI

_stream_client = OpenAI(
    api_key="ollama",
    base_url=OLLAMA_API_URL.rstrip("/") + "/v1",
    timeout=OLLAMA_TIMEOUT,
    max_retries=0,
)


def stream_chat(model: str, messages: list, **kwargs) -> str:
    """Chat completion via streaming; prints tokens live to the container log."""
    name = model.split(":", 1)[1] if model.startswith("openai:") else model
    stream = _stream_client.chat.completions.create(
        model=name, messages=messages, stream=True, **kwargs
    )
    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            parts.append(piece)
            print(piece, end="", flush=True)
    print()
    return strip_think("".join(parts))


# === Research Agent ===
def research_agent(
    prompt: str, model: str = DEFAULT_MODEL, return_messages: bool = False
):
    print("==================================")
    print("🔍 Research Agent")
    print("==================================")

    full_prompt = f"""
You are an advanced research assistant with expertise in information retrieval and academic research methodology. Your mission is to gather comprehensive, accurate, and relevant information on any topic requested by the user.

## AVAILABLE RESEARCH TOOLS:

1. **`tavily_search_tool`**: General web search engine
   - USE FOR: Recent news, current events, blogs, websites, industry reports, and non-academic sources
   - BEST FOR: Up-to-date information, diverse perspectives, practical applications, and general knowledge

2. **`arxiv_search_tool`**: Academic publication database
   - USE FOR: Peer-reviewed research papers, technical reports, and scholarly articles
   - LIMITED TO THESE DOMAINS ONLY:
     * Computer Science
     * Mathematics
     * Physics
     * Statistics
     * Quantitative Biology
     * Quantitative Finance
     * Electrical Engineering and Systems Science
     * Economics
   - BEST FOR: Scientific evidence, theoretical frameworks, and technical details in supported fields

3. **`wikipedia_search_tool`**: Encyclopedia resource
   - USE FOR: Background information, definitions, overviews, historical context
   - BEST FOR: Establishing foundational knowledge and understanding basic concepts

4. **`optica_open_access_search`**: Open access articles from Optica Publishing Group
   (Optics Express, Optica, Biomedical Optics Express, Optical Materials Express, ...)
   - USE FOR: Optics, photonics, lasers, imaging, spectroscopy, metamaterials, optical communications
   - Returns title, authors, journal, year, DOI, free PDF link, license and abstract
   - Use English keywords; you may pass `from_year` to focus on recent work

5. **`site_search`**: Web search restricted to the user's preferred websites
   - USE FOR: Finding pages on specific sites (default: the user's configured research sites)
   - You may pass `domains`, e.g. ["opg.optica.org", "spie.org"]

6. **`check_open_access`**: Given a DOI, finds a free and legal open access version (Unpaywall)
   - USE FOR: Any relevant paper found elsewhere, to record where it can be read for free

## RESEARCH METHODOLOGY:

1. **Analyze Request**: Identify the core research questions and knowledge domains
2. **Plan Search Strategy**: Determine which tools are most appropriate for the topic
3. **Execute Searches**: Use the selected tools with effective keywords and queries
4. **Evaluate Sources**: Prioritize credibility, relevance, recency, and diversity
5. **Synthesize Findings**: Organize information logically with clear source attribution
6. **Document Search Process**: Note which tools were used and why

## TOOL SELECTION GUIDELINES:

- For scientific/academic questions in supported domains → Use `arxiv_search_tool`
- For recent developments, news, or practical information → Use `tavily_search_tool`
- For fundamental concepts or historical context → Use `wikipedia_search_tool`
- For optics/photonics topics → ALWAYS use `optica_open_access_search`, and prefer open access sources
- For papers with a DOI but no free link → Use `check_open_access`
- For comprehensive research → Use multiple tools strategically
- NEVER use `arxiv_search_tool` for domains outside its supported list
- ALWAYS verify information across multiple sources when possible

## OUTPUT FORMAT:

Present your research findings in a structured format that includes:
1. **Summary of Research Approach**: Tools used and search strategy
2. **Key Findings**: Organized by subtopic or source
3. **Source Details**: Include URLs, titles, authors, and publication dates
4. **Limitations**: Note any gaps in available information

Today is {datetime.now().strftime("%Y-%m-%d")}.

USER RESEARCH REQUEST:
{prompt}
""".strip()

    messages = [{"role": "user", "content": no_think(full_prompt)}]
    tools = [
        arxiv_search_tool,
        tavily_search_tool,
        wikipedia_search_tool,
        optica_open_access_search,
        site_search,
        check_open_access,
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_turns=5,
            temperature=0.0,  # Use deterministic output
        )

        content = strip_think(resp.choices[0].message.content or "")

        # ---- Collect tool calls from intermediate_responses and intermediate_messages
        calls = []

        # A) From intermediate_responses
        for ir in getattr(resp, "intermediate_responses", []) or []:
            try:
                tcs = ir.choices[0].message.tool_calls or []
                for tc in tcs:
                    calls.append((tc.function.name, tc.function.arguments))
            except Exception:
                pass

        # B) From intermediate_messages on the final message
        for msg in getattr(resp.choices[0].message, "intermediate_messages", []) or []:
            # assistant message with tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append((tc.function.name, tc.function.arguments))

        # Dedup while preserving order
        seen = set()
        dedup_calls = []
        for name, args in calls:
            key = (name, args)
            if key not in seen:
                seen.add(key)
                dedup_calls.append((name, args))

        # Pretty print args: JSON->dict if possible
        tool_lines = []
        for name, args in dedup_calls:
            arg_text = str(args)
            try:
                import json as _json

                parsed = _json.loads(args) if isinstance(args, str) else args
                if isinstance(parsed, dict):
                    kv = ", ".join(f"{k}={repr(v)}" for k, v in parsed.items())
                    arg_text = kv
            except Exception:
                # keep raw string if not JSON
                pass
            tool_lines.append(f"- {name}({arg_text})")

        if tool_lines:
            tools_html = (
                "<h2 style='font-size:1.5em; color:#2563eb;'>📎 Tools used</h2>"
            )
            tools_html += (
                "<ul>" + "".join(f"<li>{line}</li>" for line in tool_lines) + "</ul>"
            )
            content += "\n\n" + tools_html

        print("✅ Output:\n", content)
        return content, messages

    except Exception as e:
        print("❌ Error:", e)
        return f"[Model Error: {str(e)}]", messages


def writer_agent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    min_words_total: int = 2400,
    min_words_per_section: int = 400,
    max_tokens: int = WRITER_MAX_TOKENS,
    retries: int = 1,
):
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")

    system_message = """
You are an expert academic writer with a PhD-level understanding of scholarly communication. Your task is to synthesize research materials into a comprehensive, well-structured academic report.

## REPORT REQUIREMENTS:
- Produce a COMPLETE, POLISHED, and PUBLICATION-READY academic report in Markdown format
- Create original content that thoroughly analyzes the provided research materials
- DO NOT merely summarize the sources; develop a cohesive narrative with critical analysis
- Length should be appropriate to thoroughly cover the topic (typically 1500-3000 words)

## MANDATORY STRUCTURE:
1. **Title**: Clear, concise, and descriptive of the content
2. **Abstract**: Brief summary (100-150 words) of the report's purpose, methods, and key findings
3. **Introduction**: Present the topic, research question/problem, significance, and outline of the report
4. **Background/Literature Review**: Contextualize the topic within existing scholarship
5. **Methodology**: If applicable, describe research methods, data collection, and analytical approaches
6. **Key Findings/Results**: Present the primary outcomes and evidence
7. **Discussion**: Interpret findings, address implications, limitations, and connections to broader field
8. **Conclusion**: Synthesize main points and suggest directions for future research
9. **References**: Complete list of all cited works

## ACADEMIC WRITING GUIDELINES:
- Maintain formal, precise, and objective language throughout
- Use discipline-appropriate terminology and concepts
- Support all claims with evidence and reasoning
- Develop logical flow between ideas, paragraphs, and sections
- Include relevant examples, case studies, data, or equations to strengthen arguments
- Address potential counterarguments and limitations

## CITATION AND REFERENCE RULES:
- Use numeric inline citations [1], [2], etc. for all borrowed ideas and information
- Every claim based on external sources MUST have a citation
- Each inline citation must correspond to a complete entry in the References section
- Every reference listed must be cited at least once in the text
- Preserve ALL original URLs, DOIs, and bibliographic information from source materials
- Format references consistently according to academic standards

## FORMATTING GUIDELINES:
- Use Markdown syntax for all formatting (headings, emphasis, lists, etc.)
- Include appropriate section headings and subheadings to organize content
- Format any equations, tables, or figures according to academic conventions
- Use bullet points or numbered lists when appropriate for clarity
- Use html syntax to handle all links with target="_blank", so user can always open link in new tab on both html and markdown format

Output the complete report in Markdown format only. Do not include meta-commentary about the writing process.

INTERNAL CHECKLIST (DO NOT INCLUDE IN OUTPUT):
- [ ] Incorporated all provided research materials
- [ ] Developed original analysis beyond mere summarization
- [ ] Included all mandatory sections with appropriate content
- [ ] Used proper inline citations for all borrowed content
- [ ] Created complete References section with all cited sources
- [ ] Maintained academic tone and language throughout
- [ ] Ensured logical flow and coherent structure
- [ ] Preserved all source URLs and bibliographic information
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": no_think(prompt)},
    ]

    def _call(messages_):
        return stream_chat(model, messages_, temperature=0, max_tokens=max_tokens)

    def _word_count(md_text: str) -> int:
        import re

        words = re.findall(r"\b\w+\b", md_text)
        return len(words)

    content = _call(messages)

    print("✅ Output:\n", content)
    return content, messages


def editor_agent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    target_min_words: int = 2400,
):
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")

    system_message = """
You are a professional academic editor with expertise in improving scholarly writing across disciplines. Your task is to refine and elevate the quality of the academic text provided.

## Your Editing Process:
1. Analyze the overall structure, argument flow, and coherence of the text
2. Ensure logical progression of ideas with clear topic sentences and transitions between paragraphs
3. Improve clarity, precision, and conciseness of language while maintaining academic tone
4. Verify technical accuracy (to the extent possible based on context)
5. Enhance readability through appropriate formatting and organization

## Specific Elements to Address:
- Strengthen thesis statements and main arguments
- Clarify complex concepts with additional explanations or examples where needed
- Add relevant equations, diagrams, or illustrations (described in markdown) when they would enhance understanding
- Ensure proper integration of evidence and maintain academic rigor
- Standardize terminology and eliminate redundancies
- Improve sentence variety and paragraph structure
- Preserve all citations [1], [2], etc., and maintain the integrity of the References section

## Formatting Guidelines:
- Use markdown formatting consistently for headings, emphasis, lists, etc.
- Structure content with appropriate section headings and subheadings
- Format equations, tables, and figures according to academic standards

Return only the revised, polished text in Markdown format without explanatory comments about your edits.
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": no_think(prompt)},
    ]

    content = stream_chat(model, messages, temperature=0)
    print("✅ Output:\n", content)
    return content, messages
