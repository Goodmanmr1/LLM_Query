# streamlit_app.py
# Bright Data MCP → Streamlit, now with Batch & Pipeline (chaining) modes.

import os
import io
import json
import time
import typing as T
import csv
import requests
import streamlit as st

# ------------------------------ Config helpers ------------------------------

def get_secret(name: str, default: T.Optional[str] = None) -> T.Optional[str]:
    try:
        return st.secrets.get(name, None) or os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)

BRIGHT_API = "https://api.brightdata.com"
ALL_ENGINES = ["chatgpt", "perplexity", "gemini", "google_ai", "copilot", "grok"]

def get_datasets_map() -> dict:
    return {
        "chatgpt":    get_secret("BD_DATASET_CHATGPT"),
        "perplexity": get_secret("BD_DATASET_PERPLEXITY"),
        "gemini":     get_secret("BD_DATASET_GEMINI"),
        "google_ai":  get_secret("BD_DATASET_GOOGLE_AI"),
        "copilot":    get_secret("BD_DATASET_COPILOT"),
        "grok":       get_secret("BD_DATASET_GROK"),
    }

def get_default_urls() -> dict:
    return {
        "chatgpt":    get_secret("BD_URL_CHATGPT")    or "https://chatgpt.com/",
        "perplexity": get_secret("BD_URL_PERPLEXITY") or "https://www.perplexity.ai/",
        "gemini":     get_secret("BD_URL_GEMINI")     or "https://gemini.google.com/",
        "google_ai":  get_secret("BD_URL_GOOGLE_AI")  or "https://google.com/aimode/",
        "copilot":    get_secret("BD_URL_COPILOT")    or "https://copilot.microsoft.com/",
        "grok":       get_secret("BD_URL_GROK")       or "https://x.com/i/grok",
    }

ENGINE_FIELDS = {
  "google_ai": [
    "url","prompt","answer_html","answer_text","links_attached","citations","index","country",
    "answer_text_markdown","answer_section_html","timestamp"
  ],
  "chatgpt": [
    "url","prompt","answer_html","answer_text","links_attached","citations","recommendations","country",
    "sources_not_required","is_map","references","shopping","shopping_visible","index","answer_text_markdown",
    "web_search","web_search_triggered","require_sources","additional_prompt","additional_answer_text","map",
    "search_sources","response_raw","answer_section_html","model","web_search_query","search_model_queries","timestamp"
  ],
  "perplexity": [
    "url","prompt","answer_html","answer_text","answer_text_markdown","sources","country","source_html",
    "is_shopping_data","shopping_data","index","response_raw","answer_section_html","exported_markdown",
    "export_markdown_file","related_prompts","citations","web_search_query","timestamp"
  ],
  "gemini": [
    "url","prompt","answer_html","answer_text","links_attached","citations","country","sources_not_required","index",
    "answer_text_markdown","web_search_triggered","additional_prompt","response_raw","answer_section_html","model","timestamp"
  ],
  "copilot": [
    "url","prompt","answer_text","answer_html","answer_text_markdown","sources","country","index","response_raw","timestamp"
  ],
  "grok": [
    "url","prompt","answer_text","answer_markdown","sources","citations","timestamp","index"
  ]
}

COMPACT_FIELDS = {
  "google_ai": ["url","prompt","answer_text","citations","timestamp"],
  "chatgpt":   ["url","prompt","answer_text","citations","web_search_triggered","model","timestamp"],
  "perplexity":["url","prompt","answer_text","sources","citations","timestamp"],
  "gemini":    ["url","prompt","answer_text","citations","model","timestamp"],
  "copilot":   ["url","prompt","answer_text","sources","timestamp"],
  "grok":      ["url","prompt","answer_text","sources","timestamp"],
}
MINIMAL_FIELDS = ["url","prompt","answer_text","timestamp"]

# ------------------------------ Sanitization ------------------------------

def _truncate(s: T.Any, max_chars: int) -> T.Any:
    if isinstance(s, str) and len(s) > max_chars:
        return s[:max_chars] + f"… [truncated {len(s) - max_chars} chars]"
    return s

def sanitize_object(obj: dict, fields: T.List[str], max_chars: int, max_citations: int, max_array_items: int) -> dict:
    out = {}
    for k in fields:
        if k in obj and obj[k] is not None:
            out[k] = obj[k]

    citation_fields = {"citations","sources","links_attached","references","search_sources"}
    other_array_fields = {"related_prompts","recommendations","shopping","shopping_data","search_model_queries"}

    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = _truncate(v, max_chars)
        elif isinstance(v, list):
            limit = len(v)
            if k in citation_fields and isinstance(max_citations, int):
                limit = max_citations
            elif k in other_array_fields and isinstance(max_array_items, int):
                limit = max_array_items
            v2 = v[:limit]
            def _proc(x):
                if isinstance(x, str): return _truncate(x, max_chars)
                if isinstance(x, dict):
                    c = {}
                    for kk, vv in x.items():
                        if isinstance(vv, str): c[kk] = _truncate(vv, max_chars)
                        elif isinstance(vv, list): c[kk] = [ _truncate(i, max_chars) if isinstance(i, str) else i for i in vv ]
                        else: c[kk] = vv
                    return c
                return x
            out[k] = [ _proc(x) for x in v2 ]
        elif isinstance(v, dict):
            c = {}
            for kk, vv in v.items():
                if isinstance(vv, str): c[kk] = _truncate(vv, max_chars)
                elif isinstance(vv, list): c[kk] = [ _truncate(i, max_chars) if isinstance(i, str) else i for i in vv ]
                else: c[kk] = vv
            out[k] = c
    return out

# ------------------------------ Bright Data API ------------------------------

def trigger_dataset(api_key: str, dataset_id: str, row: dict, include_errors: bool) -> dict:
    params = {"dataset_id": str(dataset_id), "include_errors": "true" if include_errors else "false"}
    url = f"{BRIGHT_API}/datasets/v3/trigger"
    resp = requests.post(url, params=params, headers={"Authorization": f"Bearer {api_key}"}, json=[row], timeout=60)
    ct = resp.headers.get("content-type","")
    payload = resp.json() if "application/json" in ct else {"raw": resp.text}
    if not resp.ok:
        raise RuntimeError(f"Trigger failed (HTTP {resp.status_code}): {json.dumps(payload, indent=2)[:2000]}")
    return payload

def poll_progress(api_key: str, snapshot_id: str, timeout_s: int = 120, interval_s: float = 3.0) -> str:
    start = time.time()
    status = "running"
    while time.time() - start < timeout_s:
        r = requests.get(f"{BRIGHT_API}/datasets/v3/progress/{snapshot_id}",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        if not r.ok:
            time.sleep(interval_s)
            continue
        status = (r.json() or {}).get("status", status)
        if status == "ready":
            return status
        if status in ("failed","error"):
            raise RuntimeError(f"Snapshot {snapshot_id} failed with status: {status}")
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for snapshot {snapshot_id}. Last status: {status}")

def download_snapshot(api_key: str, snapshot_id: str) -> T.List[dict]:
    r = requests.get(f"{BRIGHT_API}/datasets/v3/snapshot/{snapshot_id}",
                     params={"format":"json"},
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else [data]

def run_ai_search(api_key: str,
                  engine: str,
                  dataset_id: str,
                  url: str,
                  prompt: str,
                  index: int = 1,
                  country: T.Optional[str] = None,
                  web_search: T.Optional[bool] = None,
                  require_sources: T.Optional[bool] = None,
                  additional_prompt: T.Optional[str] = None,
                  extra: T.Optional[dict] = None,
                  include_errors: bool = False,
                  return_snapshot_link: bool = False) -> dict:
    row = {"url": url, "prompt": prompt, "index": index}
    if country: row["country"] = country
    if web_search is not None: row["web_search"] = bool(web_search)
    if require_sources is not None: row["require_sources"] = bool(require_sources)
    if additional_prompt: row["additional_prompt"] = additional_prompt
    if extra: row.update(extra)

    payload = trigger_dataset(api_key, dataset_id, row, include_errors)
    snap_id = payload.get("snapshot_id")

    if return_snapshot_link or not snap_id:
        return {"trigger_response": payload, "snapshot_id": snap_id}

    poll_progress(api_key, snap_id)
    rows = download_snapshot(api_key, snap_id)
    return {"snapshot_id": snap_id, "rows": rows}

# ------------------------------ Flatteners & Export ------------------------------

def flatten_for_csv(record: dict) -> dict:
    row = {}
    for k, v in record.items():
        if isinstance(v, (str, int, float)) or v is None:
            row[k] = v
        else:
            try:
                row[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                row[k] = str(v)
    return row

def to_markdown_table(rows: T.List[dict], fields: T.List[str]) -> str:
    if not rows:
        return "_No data_"
    md = io.StringIO()
    md.write("| " + " | ".join(fields) + " |\n")
    md.write("|" + "|".join(["---"]*len(fields)) + "|\n")
    for r in rows:
        vals = []
        for f in fields:
            v = r.get(f, "")
            if isinstance(v, (dict, list)):
                v = "```json\n" + json.dumps(v, ensure_ascii=False, indent=2) + "\n```"
            else:
                v = str(v).replace("\n", " ").replace("|", "\\|")
            vals.append(v)
        md.write("| " + " | ".join(vals) + " |\n")
    return md.getvalue()

def build_export_bytes(data_by_engine: dict, fields_by_engine: dict, export_type: str) -> bytes:
    export_type = export_type.lower()
    if export_type == "json":
        payload = {"engines": list(data_by_engine.keys()), "data": data_by_engine, "fields": fields_by_engine}
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    elif export_type == "csv":
        all_fields = set(["engine"])
        for fields in fields_by_engine.values():
            all_fields.update(fields)
        # also capture run_label / chain_step if present
        all_fields.update(["run_label","chain_step"])
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(all_fields), extrasaction="ignore")
        writer.writeheader()
        for eng, rows in data_by_engine.items():
            for r in rows:
                row = flatten_for_csv(r)
                row["engine"] = eng
                writer.writerow(row)
        return buf.getvalue().encode("utf-8")
    elif export_type == "markdown":
        md = io.StringIO()
        for eng in data_by_engine:
            rows = data_by_engine[eng]
            fields = fields_by_engine[eng] + [f for f in ["run_label","chain_step"] if any(f in r for r in rows)]
            md.write(f"## {eng}\n\n")
            if not rows:
                md.write("_No data_\n\n")
                continue
            md.write(to_markdown_table(rows, fields))
            md.write("\n\n")
        return md.getvalue().encode("utf-8")
    else:
        raise ValueError(f"Unsupported export type: {export_type}")

# ------------------------------ UI ------------------------------

st.set_page_config(page_title="Bright Data AI Search (Streamlit)", layout="wide")
st.title("🔎 Bright Data → AI Search (Streamlit)")

with st.sidebar:
    st.header("Configuration")
    api_key = get_secret("BRIGHT_DATA_API_KEY") or st.text_input("BRIGHT_DATA_API_KEY", type="password")
    st.caption("Tip: Prefer **st.secrets** when deploying. Locally, you can set env vars or a .env file.")

    datasets_map = get_datasets_map()
    default_urls = get_default_urls()

    run_mode = st.radio("Run mode", ["Single", "Batch (multi-prompts)", "Pipeline (chain engines)"], index=0)
    engines_default = ["chatgpt"] if run_mode != "Pipeline (chain engines)" else ["chatgpt","perplexity"]
    engines = st.multiselect("Engines", ALL_ENGINES, default=engines_default)

    with st.expander("Advanced: per-engine dataset overrides (optional)"):
        overrides = {}
        for eng in ALL_ENGINES:
            txt = st.text_input(f"{eng} dataset id (overrides secrets)", key=f"override_ds_{eng}", value="")
            if txt.strip():
                overrides[eng] = txt.strip()

    st.divider()
    st.subheader("Request")

    if run_mode == "Single":
        prompt = st.text_area("Prompt", placeholder="Ask the engine(s) something…", height=140)
    elif run_mode == "Batch (multi-prompts)":
        prompts_multiline = st.text_area("Prompts (one per line). Optional per-line URL using ' | ' separator: prompt | https://url", height=160)
        prompt = ""  # not used
    else:
        start_prompt = st.text_area("Pipeline: starting prompt", height=120, placeholder="Initial question…")
        chain_template = st.text_area("Template for subsequent engines (use {prev} to insert previous answer_text)", height=140,
                                      value="Using the following context, refine or add citations:\n\n{prev}\n\nTask: Provide a concise, cited answer.")
        max_prev_chars = st.slider("Max chars from previous answer to insert", 200, 8000, 2500, step=100)

    col1, col2 = st.columns(2)
    with col1:
        index = st.number_input("Index (row)", min_value=1, value=1, step=1)
        country = st.text_input("Country (optional, e.g., us, uk, de)", value="")
        additional_prompt = st.text_area("[ChatGPT/Gemini] Additional prompt (optional)", value="", height=80)
    with col2:
        web_search = st.checkbox("[ChatGPT] Force web search", value=False)
        require_sources = st.checkbox("[ChatGPT] Require sources", value=False)
        entry_url = st.text_input("Entry URL (applies to all engines unless engine has a default)", value="")

    extra_json_str = st.text_area("Extra JSON (engine-specific, merged into row)", placeholder='{"some_flag": true}', height=90)

    st.divider()
    st.subheader("Response shaping")
    preset = st.radio("Fields preset", ["Full (engine schema)","Compact","Custom"], index=0, horizontal=True)
    exclude_html = st.checkbox("Exclude HTML fields", value=False)
    exclude_raw = st.checkbox("Exclude response_raw", value=True)
    include_errors = st.checkbox("Include error/warning fields", value=False)
    return_snapshot_link = st.checkbox("Return snapshot link only", value=False)

    max_items = st.slider("Max rows per engine", 1, 25, 1)
    max_chars = st.slider("Max chars per field", 200, 8000, 4000, step=100)
    max_citations = st.slider("Max citations/sources items", 0, 50, 10)
    max_array_items = st.slider("Max other array items", 0, 20, 5)

    export_type = st.radio("Export type", ["JSON","CSV","Markdown"], index=0, horizontal=True)

    custom_fields = {}
    if preset == "Custom":
        st.write("Select fields to keep per engine:")
        for eng in ALL_ENGINES:
            candidates = ENGINE_FIELDS.get(eng, MINIMAL_FIELDS)
            custom_fields[eng] = st.multiselect(f"{eng} fields", options=candidates, default=candidates)

run = st.button("Run")

def resolve_dataset(eng: str, overrides: dict, datasets_map: dict) -> T.Optional[str]:
    if eng in overrides and overrides[eng]:
        return overrides[eng]
    return datasets_map.get(eng)

def effective_fields_for_engine(eng: str) -> T.List[str]:
    if preset == "Custom" and custom_fields.get(eng):
        eff = list(custom_fields[eng])
    elif preset == "Compact":
        eff = COMPACT_FIELDS.get(eng, MINIMAL_FIELDS)
    else:
        eff = ENGINE_FIELDS.get(eng, MINIMAL_FIELDS)[:]
    if exclude_html:
        eff = [f for f in eff if ("html" not in f and "_html" not in f)]
    if exclude_raw:
        eff = [f for f in eff if f != "response_raw"]
    if include_errors:
        for f in ["error","error_code","warning","warning_code","input"]:
            if f not in eff:
                eff.append(f)
    # Always include run metadata if present
    for meta in ["run_label","chain_step"]:
        if meta not in eff:
            eff.append(meta)
    return eff

if run:
    if not api_key:
        st.error("Missing BRIGHT_DATA_API_KEY. Add it to Streamlit secrets or environment.")
        st.stop()
    if not engines:
        st.error("Pick at least one engine.")
        st.stop()

    extra = None
    if extra_json_str.strip():
        try:
            extra = json.loads(extra_json_str)
        except Exception as e:
            st.warning(f"Ignoring Extra JSON: {e}")

    data_by_engine: dict = {eng: [] for eng in engines}
    fields_by_engine: dict = {}
    missing = []

    def run_once(eng: str, url: str, prompt_text: str, run_label: str, chain_step: int = 0):
        ds = resolve_dataset(eng, overrides, datasets_map)
        if not ds:
            missing.append(eng)
            fields_by_engine.setdefault(eng, effective_fields_for_engine(eng))
            return

        try:
            res = run_ai_search(
                api_key=api_key,
                engine=eng,
                dataset_id=ds,
                url=url,
                prompt=prompt_text,
                index=int(index),
                country=country or None,
                web_search=web_search if eng == "chatgpt" else None,
                require_sources=require_sources if eng == "chatgpt" else None,
                additional_prompt=additional_prompt or None,
                extra=extra,
                include_errors=include_errors,
                return_snapshot_link=return_snapshot_link
            )
            if return_snapshot_link or "rows" not in res:
                row = {"snapshot_id": res.get("snapshot_id"), "trigger_response": res.get("trigger_response"),
                       "run_label": run_label, "chain_step": chain_step}
                data_by_engine[eng].append(row)
                fields_by_engine.setdefault(eng, ["snapshot_id","trigger_response","run_label","chain_step"])
                return None
            else:
                rows = res.get("rows", [])
                eff = effective_fields_for_engine(eng)
                pruned = [
                    sanitize_object(r, eff, max_chars=max_chars, max_citations=max_citations, max_array_items=max_array_items)
                    for r in rows[:max_items]
                ]
                # attach metadata
                for r in pruned:
                    r["run_label"] = run_label
                    r["chain_step"] = chain_step
                data_by_engine[eng].extend(pruned)
                fields_by_engine.setdefault(eng, eff)
                # return a concatenated previous text for chaining convenience
                if pruned:
                    # prefer answer_text, fallback to answer_text_markdown, else stringify first row
                    prev = pruned[0].get("answer_text") or pruned[0].get("answer_text_markdown") or json.dumps(pruned[0])
                    return str(prev)
                return None
        except Exception as e:
            st.error(f"{eng} → {type(e).__name__}: {e}")
            fields_by_engine.setdefault(eng, effective_fields_for_engine(eng))
            return None

    progress = st.progress(0.0, text="Starting…")

    if run_mode == "Single":
        if not st.session_state.get("single_prompt_cache"):
            st.session_state["single_prompt_cache"] = ""
        if not prompt.strip():
            st.error("Please enter a prompt.")
            st.stop()
        for idx, eng in enumerate(engines, start=1):
            progress.progress((idx-1)/max(len(engines),1), text=f"Running {eng}…")
            url = (entry_url.strip() or get_default_urls().get(eng, ""))
            run_once(eng, url, prompt, run_label="single", chain_step=0)
            progress.progress(idx/max(len(engines),1), text=f"Finished {eng}")

    elif run_mode == "Batch (multi-prompts)":
        lines = [ln.strip() for ln in (prompts_multiline or "").splitlines() if ln.strip()]
        if not lines:
            st.error("Enter at least one prompt line.")
            st.stop()
        # Format: "prompt | optional_url"
        parsed = []
        for ln in lines:
            if " | " in ln:
                p, u = ln.split(" | ", 1)
                parsed.append((p.strip(), u.strip()))
            else:
                parsed.append((ln, ""))
        total = len(parsed) * max(1, len(engines))
        done = 0
        for i, (ptext, maybe_url) in enumerate(parsed, start=1):
            for eng in engines:
                done += 1
                progress.progress(done/max(total,1), text=f"Running {eng} (batch {i}/{len(parsed)})…")
                url = (maybe_url or entry_url.strip() or get_default_urls().get(eng, ""))
                run_once(eng, url, ptext, run_label=f"batch-{i}", chain_step=0)

    else:  # Pipeline (chain engines)
        if not start_prompt.strip():
            st.error("Enter a starting prompt for the pipeline.")
            st.stop()
        # We execute engines sequentially; each step sees previous step's text injected into {prev}
        prev_text = ""
        for idx, eng in enumerate(engines, start=1):
            progress.progress((idx-1)/max(len(engines),1), text=f"Pipeline step {idx}: {eng}…")
            url = (entry_url.strip() or get_default_urls().get(eng, ""))
            prompt_text = start_prompt if idx == 1 else chain_template.replace("{prev}", (prev_text or "")[:max_prev_chars])
            prev_text = run_once(eng, url, prompt_text, run_label="pipeline", chain_step=idx) or prev_text
            progress.progress(idx/max(len(engines),1), text=f"Finished {eng}")

    if missing:
        st.warning("No dataset ID found in secrets for: " + ", ".join(sorted(set(missing))) + ". "
                   "Add them to Streamlit secrets (e.g., BD_DATASET_CHATGPT) or provide overrides in the expander.")

    try:
        blob = build_export_bytes(data_by_engine, fields_by_engine, export_type)
        fname = f"bright_ai_search.{export_type.lower()}"
        st.download_button(f"Download {export_type}", data=blob, file_name=fname,
                           mime=("text/markdown" if export_type.lower()=="markdown" else
                                 "text/csv" if export_type.lower()=="csv" else "application/json"))
    except Exception as e:
        st.error(f"Export error: {type(e).__name__}: {e}")

    st.subheader("Preview")
    with st.expander("Data by engine"):
        st.json({"data": data_by_engine, "fields": fields_by_engine}, expanded=False)

with st.expander("Environment status (read-only)"):
    datasets_map = get_datasets_map()
    st.write({
        "dataset_ids_loaded": {k: bool(v) for k, v in datasets_map.items()},
        "default_urls": get_default_urls(),
    })

st.caption("Run multiple engines, batches of prompts, or pipeline across engines with previous answers feeding the next step.")
