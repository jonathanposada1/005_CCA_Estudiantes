import os
import json
from datetime import datetime
from urllib import response
from aisuite import Client
##from src.research_tools import (
##    arxiv_search_tool,
##    tavily_search_tool,
##    wikipedia_search_tool,
##)

##ACA ESTA LA MODIFICACION PARA EL ADS

from src.research_tools import (
    arxiv_search_tool,
    arxiv_tool_def,
    tavily_search_tool,
    tavily_tool_def,
    wikipedia_search_tool,
    wikipedia_tool_def,
    ads_search_tool,
    ads_tool_def,
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

client = Client(
    provider_configs={
        "openai": {
            "api_key": "ollama",  # Ollama ignores it, but the OpenAI SDK requires a value
            "base_url": OLLAMA_API_URL.rstrip("/") + "/v1",
            "timeout": OLLAMA_TIMEOUT,
        }
    }
)


# === Research Agent ===
def research_agent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    return_messages: bool = False
):
    print("==================================")
    print("🔍 Research Agent")
    print("==================================")

    full_prompt = f"""
You are an advanced research assistant with expertise in information retrieval
and academic research methodology.

IMPORTANT RULE:
When using any research tool, especially ads_search_tool, you MUST ONLY
report information that was actually returned by the tool.

NEVER invent papers, authors, titles, DOIs, dates, numerical results,
or abstracts from your own knowledge.

If ads_search_tool returns papers, use ONLY those papers in your answer.
If the tool returns no papers, say that no papers were found.


Your task is to execute the research step requested below.

IMPORTANT:
- You MUST actually use the appropriate search tool when the task requires it.
- Do NOT pretend that you searched a database.
- Do NOT invent search results.
- If a tool is available, call the tool.
- After receiving the tool result, analyze it and produce the requested output.

AVAILABLE TOOLS:

1. tavily_search_tool
   General web search.

2. arxiv_search_tool
   Search academic preprints on arXiv.

3. wikipedia_search_tool
   Search Wikipedia for background information.

4. ads_search_tool
   Search the NASA Astrophysics Data System (NASA ADS).
   Use this for astronomy, astrophysics, cosmology and related literature.

RESEARCH STEP TO EXECUTE:
{prompt}

Return the research findings based ONLY on information actually obtained
from the tools or information explicitly provided in the prompt.
""".strip()

    messages = [
        {"role": "user", "content": full_prompt}
    ]

    # ---------------------------------------------------------
    # Available Python tools
    # ---------------------------------------------------------

    available_tools = {
        "tavily_search_tool": tavily_search_tool,
        "arxiv_search_tool": arxiv_search_tool,
        "wikipedia_search_tool": wikipedia_search_tool,
        "ads_search_tool": ads_search_tool,
    }

    tools = [
        arxiv_search_tool,
        tavily_search_tool,
        wikipedia_search_tool,
        ads_search_tool,
    ]


    # ---------------------------------------------------------
    # Decide whether this research step requires a specific tool
    # ---------------------------------------------------------

    prompt_lower = prompt.lower()

    forced_tool = None

    if "nasa ads" in prompt_lower or "ads_search_tool" in prompt_lower:
        forced_tool = "ads_search_tool"

    elif "arxiv" in prompt_lower:
        forced_tool = "arxiv_search_tool"

    elif "tavily" in prompt_lower:
        forced_tool = "tavily_search_tool"

    # ---------------------------------------------------------
    # First LLM call
    # ---------------------------------------------------------

    try:

        if forced_tool:
            print(f"🛠️ Forced tool: {forced_tool}")

            if forced_tool == "ads_search_tool":
                print("🔭 Ejecutando ADS directamente...")

                result = ads_search_tool(
                    "dark matter subhalos Milky Way",
                    max_results=5
                )

                print("📦 RESULTADO ADS:")
                print(result)

                messages.append({
                    "role": "user",
                    "content": (
                        "Estos son los resultados REALES obtenidos de NASA ADS. "
                        "Usa EXCLUSIVAMENTE estos resultados para responder. "
                        "NO inventes ningún paper.\n\n"
                        + json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str
                        )
                    )
                })

                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                )

                content = resp.choices[0].message.content or ""

                print("ℹ️ ADS ejecutado directamente.")
                print("✅ Final research output:")
                print(content)

                return content, messages

            tool_choice = {
                "type": "function",
                "function": {
                    "name": forced_tool
                }
            }

        else:
            tool_choice = "auto"

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.0,
        )

        # -----------------------------------------------------
        # MANUAL TOOL-CALL LOOP
        # -----------------------------------------------------

        max_tool_calls = 5

        for iteration in range(max_tool_calls):

            message = resp.choices[0].message

            tool_calls = getattr(message, "tool_calls", None)
            
            print("🔎 TOOL CALLS:", tool_calls)

            # ---------------------------------------------
            # No tool call -> final answer
            # ---------------------------------------------

            if not tool_calls:
                content = message.content or ""

                print("ℹ️ No more tool calls.")
                print("✅ Final research output:")
                print(content)

                return content, messages

            # ---------------------------------------------
            # Add assistant message containing tool call
            # ---------------------------------------------

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_calls,
            })

            # ---------------------------------------------
            # Execute every requested tool
            # ---------------------------------------------

            for tool_call in tool_calls:

                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                print("\n" + "=" * 60)
                print(f"🔧 TOOL CALL #{iteration + 1}")
                print(f"Tool: {tool_name}")
                print(f"Arguments: {raw_arguments}")
                print("=" * 60)

                # -----------------------------------------
                # Check tool exists
                # -----------------------------------------

                if tool_name not in available_tools:

                    tool_result = {
                        "error": f"Unknown tool: {tool_name}"
                    }

                else:

                    tool_function = available_tools[tool_name]

                    # -------------------------------------
                    # Parse arguments
                    # -------------------------------------

                    try:
                        arguments = json.loads(raw_arguments)
                    except Exception as e:

                        print(f"❌ Could not parse arguments: {e}")

                        tool_result = {
                            "error": "Invalid JSON arguments",
                            "raw_arguments": raw_arguments,
                        }

                    else:

                        # ---------------------------------
                        # ACTUALLY EXECUTE THE TOOL
                        # ---------------------------------

                        try:

                            print(
                                f"🚀 Executing {tool_name}("
                                f"{arguments})"
                            )

                            result = tool_function(**arguments)

                            print("✅ Tool executed successfully.")
                            print("📦 Result:")
                            print(result)

                            tool_result = result

                        except Exception as e:

                            print(
                                f"❌ ERROR executing "
                                f"{tool_name}: {e}"
                            )

                            tool_result = {
                                "error": str(e)
                            }

                # -----------------------------------------
                # Send tool result back to the model
                # -----------------------------------------

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str
                    ),
                })

            # ---------------------------------------------
            # Ask model to continue using tool results
            # ---------------------------------------------

            print("\n🤖 Sending tool result back to Qwen...")

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )

        # -----------------------------------------------------
        # Maximum number of tool calls reached
        # -----------------------------------------------------

        content = (
            resp.choices[0].message.content
            or "[Maximum number of tool calls reached.]"
        )

        print("⚠️ Maximum tool-call iterations reached.")
        print("✅ Output:")
        print(content)

        return content, messages

    except Exception as e:

        print("❌ Research Agent Error:")
        print(e)

        return (
            f"[Research Agent Error: {str(e)}]",
            messages
        )


def writer_agent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    min_words_total: int = 2400,
    min_words_per_section: int = 400,
    max_tokens: int = 15000,
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
        {"role": "user", "content": prompt},
    ]

    def _call(messages_):
        resp = client.chat.completions.create(
            model=model,
            messages=messages_,
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

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
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )

    content = response.choices[0].message.content
    print("✅ Output:\n", content)
    return content, messages
