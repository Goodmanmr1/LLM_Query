# Bright Data AI Search — Streamlit UI

This Streamlit app mirrors your MCP sidecar (`ai-search-mcp.js`) to run Bright Data AI Scraper datasets for engines like ChatGPT, Perplexity, Gemini, Google AI, Copilot, and Grok. It triggers a run, polls progress, downloads the snapshot, and displays sanitized JSON.

## Quick start

1. **Create a virtualenv** (optional)
2. `pip install -r requirements.txt`
3. **Set secrets** (recommended): add your keys to `.streamlit/secrets.toml`:
   ```toml
   BRIGHT_DATA_API_KEY = "xxxxxxxx"
   BD_DATASET_CHATGPT = "gd_xxx"
   BD_DATASET_PERPLEXITY = "gd_xxx"
   BD_DATASET_GEMINI = "gd_xxx"
   BD_DATASET_GOOGLE_AI = "gd_xxx"
   BD_DATASET_COPILOT = "gd_xxx"
   BD_DATASET_GROK = "gd_xxx"  # optional
   ```
   Or export the same names as environment variables.

4. `streamlit run streamlit_app.py`

## Notes

- You can override the dataset ID / entry URL per engine directly in the sidebar.
- Response shaping supports the same presets as your MCP server: Full, Compact, or Custom fields; caps for text length and array sizes; and optional exclusion of `response_raw` and HTML fields.
- Errors from the Bright Data API are surfaced inline.
