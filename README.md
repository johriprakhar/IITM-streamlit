# AI Curious Mind

A generic teaching chatbot built with Streamlit. It ships **no course-specific
or proprietary content** - only a reusable "meta-prompt" that tells the model
how to teach whatever you paste in. That makes it safe to commit to a public
repository.

## What it does

Paste any of the following into the chat box:

- a single topic or term
- a section heading
- a bullet list of sub-topics
- a whole module copied straight out of a spreadsheet or curriculum doc

For teaching content, the assistant automatically produces:

- **Analogy** - a plain-language comparison
- **How it works** - the real technical depth
- **Code example** - where it's relevant
- **Business use case** - a concrete scenario
- **Why it matters** - the payoff

Plain questions (like "what is a hash map?") are answered directly without
forcing the full template.

## Providers

Works with **OpenAI** and **Anthropic (Claude)**. You supply your own API key
in the sidebar.

### API key handling

- The key is stored **only** in `st.session_state` (this browser session's
  server-side memory).
- It is **never written to disk** and **never logged** by this app.
- Streamlit session state is isolated per session, so one visitor's key is not
  visible to another.
- Closing the tab discards the key.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), pick a
provider, paste your API key in the sidebar, and start pasting content.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit app and the generic meta-prompt |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps secrets and local cruft out of git |
