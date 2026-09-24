"""
AI Curious Mind - a generic teaching chatbot (Streamlit)

This app contains NO course-specific / proprietary content. Instead it ships a
reusable "meta-prompt" that teaches the model HOW to teach whatever the learner
pastes in: a topic, a heading, a bullet list of sub-topics, or a whole module
copied out of a spreadsheet or curriculum doc.

For any pasted content the assistant automatically produces:
  - a plain-language analogy
  - the technical depth / how it actually works
  - a code example where relevant
  - a real business use case
  - why it matters

Plain questions ("what is a hash map?") work too - the assistant answers them
directly without forcing the full teaching template when it doesn't fit.

Providers: OpenAI and Anthropic (Claude).

Security model for the API key:
  - The key is entered in the sidebar and stored ONLY in st.session_state,
    which lives in this browser session's server-side memory for the session.
  - It is never written to disk and never logged by this app.
  - Streamlit session state is isolated per session, so one visitor's key is
    not visible to another visitor.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# The meta-prompt. This is the only "content" the app ships, and it is generic:
# it describes teaching behaviour, not any particular curriculum.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are "AI Curious Mind", a patient, enthusiastic teacher for technical and \
business topics. Your job is to teach whatever the learner pastes in, whether \
that is a single term, a section heading, a bullet list of sub-topics, or an \
entire module copied out of a spreadsheet or curriculum document.

HOW TO RESPOND

1. First, decide what kind of input you received:
   - A plain question (e.g. "what is a hash map?"): just answer it clearly and \
directly. Do not force the full template below onto it.
   - Teaching content: a topic, heading, list of sub-topics, or a pasted \
module. For this, teach it using the structure below.

2. When teaching content, cover these sections (use short markdown headings). \
Skip any section that genuinely does not apply, but prefer to include them:
   - **Analogy** - an everyday, plain-language comparison so a newcomer gets \
the intuition.
   - **How it works** - the real technical depth: mechanisms, definitions, \
trade-offs. Be accurate and specific.
   - **Code example** - a short, runnable example when the topic is something \
you can show in code. State the language. Skip only if code truly does not \
apply.
   - **Business use case** - a concrete, realistic scenario where this is used \
to create value.
   - **Why it matters** - the payoff: what problem it solves or what it \
unlocks.

3. If the learner pasted several sub-topics or a whole module, briefly work \
through each one so nothing is dropped, but keep the whole reply focused and \
readable. Group related items instead of repeating yourself.

STYLE
   - Be concrete over vague. Prefer real examples to hand-waving.
   - Keep it tight; do not pad. Depth where it helps, brevity everywhere else.
   - Use markdown: headings, short paragraphs, bullet lists, fenced code \
blocks with a language tag.
   - If something is ambiguous, make a reasonable assumption and say so in one \
line rather than stalling with clarifying questions.
   - Never invent facts. If you are unsure, say so.
"""

# Model choices per provider. Kept small and current; the user can also type a
# custom model name.
PROVIDER_MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "Anthropic (Claude)": [
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
    ],
}

# Approximate public API pricing in USD per 1 million tokens (input / output).
# Indicative only - always confirm current rates on the provider's pricing page.
# Sources: OpenAI (platform.openai.com/pricing) and Anthropic
# (anthropic.com pricing) as published in 2025.
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    # Anthropic (Claude)
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-3-opus-latest": {"input": 15.00, "output": 75.00},
}


def pricing_table(provider: str) -> pd.DataFrame:
    """Build a pricing table for a provider, sorted cheapest to most expensive.

    Rows are ordered by input price (then output price) so the table runs from
    the minimum-priced model to the maximum-priced one.
    """
    rows = []
    for model_name in PROVIDER_MODELS.get(provider, []):
        price = MODEL_PRICING.get(model_name)
        if not price:
            continue
        rows.append(
            {
                "Model": model_name,
                "Input ($/1M tokens)": price["input"],
                "Output ($/1M tokens)": price["output"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["Input ($/1M tokens)", "Output ($/1M tokens)"],
            ascending=True,
        ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Initialise session state keys once per session."""
    if "messages" not in st.session_state:
        # Chat transcript excluding the system prompt (that is prepended at
        # request time). Each item: {"role": "user"|"assistant", "content": str}
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------
def call_openai(api_key: str, model: str, messages: list[dict]) -> str:
    """Call the OpenAI Chat Completions API and return the assistant text."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
    )
    return response.choices[0].message.content or ""


def call_anthropic(api_key: str, model: str, messages: list[dict]) -> str:
    """Call the Anthropic Messages API and return the assistant text.

    Anthropic takes the system prompt as a separate top-level argument rather
    than as a message in the list.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=messages,
        max_tokens=4096,
    )
    # Content is a list of blocks; concatenate any text blocks.
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "".join(parts)


def get_reply(provider: str, api_key: str, model: str, messages: list[dict]) -> str:
    """Dispatch to the correct provider."""
    if provider == "OpenAI":
        return call_openai(api_key, model, messages)
    if provider == "Anthropic (Claude)":
        return call_anthropic(api_key, model, messages)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[str, str, str]:
    """Render the sidebar controls and return (provider, model, api_key)."""
    with st.sidebar:
        st.header("Settings")

        provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()))

        model_choices = PROVIDER_MODELS[provider]
        model = st.selectbox("Model", model_choices)
        custom_model = st.text_input(
            "…or custom model name",
            value="",
            placeholder="leave blank to use the selection above",
            help="Overrides the dropdown if filled in.",
        )
        if custom_model.strip():
            model = custom_model.strip()

        # Pricing reference for the selected provider, cheapest first.
        with st.expander(f"{provider} pricing (min → max)", expanded=True):
            df = pricing_table(provider)
            if df.empty:
                st.caption("No pricing data available for this provider.")
            else:
                st.dataframe(
                    df.style.format(
                        {
                            "Input ($/1M tokens)": "${:,.2f}",
                            "Output ($/1M tokens)": "${:,.2f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "USD per 1M tokens, sorted cheapest to most expensive. "
                    "Indicative only - check the provider's pricing page for "
                    "current rates."
                )

        # The key input. type="password" masks it in the UI. We store it in
        # session_state so it survives reruns without being persisted anywhere.
        api_key = st.text_input(
            f"{provider} API key",
            value=st.session_state.api_key,
            type="password",
            help=(
                "Stored only in this browser session's memory (st.session_state). "
                "Never written to disk or logged."
            ),
        )
        st.session_state.api_key = api_key

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "Your API key lives only in this session's memory and is isolated "
            "from other visitors. Closing the tab discards it."
        )

    return provider, model, api_key


def render_intro() -> None:
    st.title("AI Curious Mind by Prakhar")
    st.write(
        "Paste a topic, a heading, a bullet list of sub-topics, or a whole "
        "module and I'll teach it - with an analogy, the technical depth, a "
        "code example where it fits, a business use case, and why it matters. "
        "Plain questions work too."
    )


def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="AI Curious Mind by Prakhar Johri", page_icon="🧠", layout="centered")
    init_state()

    provider, model, api_key = render_sidebar()
    render_intro()
    render_history()

    prompt = st.chat_input("Paste content or ask a question…")
    if not prompt:
        return

    # Echo the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key.strip():
        with st.chat_message("assistant"):
            st.warning("Add your API key in the sidebar to get a response.")
        return

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = get_reply(
                    provider=provider,
                    api_key=api_key.strip(),
                    model=model,
                    messages=st.session_state.messages,
                )
            except Exception as exc:  # surface a clean message to the user
                st.error(f"Request failed: {exc}")
                # Roll back the user turn so a retry does not duplicate context.
                st.session_state.messages.pop()
                return
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
