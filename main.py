
import sys
import os
import json
import logging
import time
import uuid
from pathlib import Path

# Challenge repo root (where this script lives)
CHALLENGE_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(CHALLENGE_ROOT / ".env")
except ImportError:
    pass  # optional: run without .env in environments that don't have python-dotenv

# Cactus repo is a sibling folder → cactus/python/src, cactus/weights/...
CACTUS_ROOT = CHALLENGE_ROOT.parent / "cactus"
CACTUS_SRC = CACTUS_ROOT / "python" / "src"
sys.path.insert(0, str(CACTUS_SRC))

functiongemma_path = str(CACTUS_ROOT / "weights" / "functiongemma-270m-it")

# Printed once per benchmark run (first time generate_cactus is called)
_system_prompt_printed = False

from cactus import cactus_init, cactus_complete, cactus_destroy  # type: ignore[import-untyped]
from google import genai
from prompts import STRICT_CACTUS_SYSTEM_PROMPT
from google.genai import types
from google.genai.errors import ClientError

# ---------------------------------------------------------------------------
# Production-ready structured logging
# - Log file: human-readable, written to LOG_FILE or logs/functiongemma.log
# - Stdout: LOG_FORMAT=json for production; otherwise human-readable.
# ---------------------------------------------------------------------------
LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FORMAT_JSON = os.environ.get("LOG_FORMAT", "").lower() == "json"
_log_dir = CHALLENGE_ROOT / "logs"
_default_log_file = _log_dir / "functiongemma.log"
LOG_FILE = os.environ.get("LOG_FILE")
_log_path = Path(LOG_FILE) if LOG_FILE else _default_log_file

logger = logging.getLogger("functiongemma")
logger.setLevel(LOG_LEVEL)
logger.handlers.clear()


class _StdoutFormatter(logging.Formatter):
    """Stdout: message as-is, or JSON from record.payload when LOG_FORMAT_JSON."""

    def __init__(self, use_json: bool, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._use_json = use_json

    def format(self, record: logging.LogRecord) -> str:
        if self._use_json and getattr(record, "payload", None) is not None:
            return json.dumps(record.payload)
        return record.getMessage()


# Stdout
_stdout = logging.StreamHandler(sys.stdout)
_stdout.setLevel(LOG_LEVEL)
_stdout.setFormatter(_StdoutFormatter(LOG_FORMAT_JSON))
logger.addHandler(_stdout)

class _FileFormatter(logging.Formatter):
    """Log file: message line plus payload (input/output/expected) when present."""

    def format(self, record: logging.LogRecord) -> str:
        line = record.getMessage()
        payload = getattr(record, "payload", None)
        if payload and isinstance(payload, dict):
            # Always include input/output/expected when present so logs are self-contained
            detail = {k: v for k, v in payload.items() if k not in ("event", "ts_iso")}
            if detail:
                line += "\n  " + json.dumps(detail, indent=2).replace("\n", "\n  ")
        return line


# Log file: human-readable + payload (input, model output, expected)
_file_handler: logging.FileHandler | None = None
if _log_path:
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(_log_path, encoding="utf-8")
    _file_handler.setLevel(LOG_LEVEL)
    _file_handler.setFormatter(_FileFormatter())
    logger.addHandler(_file_handler)


def compact_messages_for_log(messages: list[dict]) -> list[dict]:
    """Compact messages for logging: role + content preview (max 500 chars per content)."""
    out = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, str) and len(content) > 500:
            content = content[:500] + "..."
        out.append({"role": role, "content": content})
    return out


def _event_summary(event: str, payload: dict) -> str:
    """One-line plain-English summary so the log is readable at a glance."""
    conf = payload.get("confidence") or payload.get("local_confidence")
    thresh = payload.get("confidence_threshold")
    src = payload.get("source")
    if event == "benchmark_case_start":
        i, total = payload.get("case_index"), payload.get("total")
        return f"Benchmark case started (case {i}/{total})"
    if event == "benchmark_case_complete":
        return f"Case finished: F1={payload.get('f1')} source={payload.get('source')} total={payload.get('total_time_ms')}ms"
    if event == "on_device_start":
        return f"Running on-device model ({payload.get('num_tools')} tools)"
    if event == "on_device_complete":
        c, n, t = payload.get("confidence"), payload.get("num_function_calls"), payload.get("total_time_ms")
        c = c if c is not None else 0
        return f"On-device done: confidence={c:.2f} {n} tool call(s) in {t:.0f}ms"
    if event == "hybrid_decision":
        c, t = (conf if conf is not None else 0), (thresh if thresh is not None else 0.50)
        if src == "on-device":
            return f"Using on-device (confidence {c:.2f} >= {t})"
        reason = payload.get("reason", "confidence_below_threshold")
        if reason == "zero_tool_calls":
            return f"Using cloud (on-device returned 0 tool calls)"
        return f"Using cloud (on-device confidence {c:.2f} < threshold {t})"
    if event == "cloud_start":
        return f"Calling Gemini cloud ({payload.get('model')}, {payload.get('num_tools')} tools)"
    if event == "cloud_complete":
        return f"Cloud done: {payload.get('num_function_calls')} call(s) in {payload.get('total_time_ms'):.0f}ms"
    if event == "cloud_fallback_failed":
        return f"Cloud failed ({payload.get('error_type')}), using on-device result"
    if event == "on_device_parse_error":
        return f"On-device response parse error: {payload.get('error', '')[:80]}"
    if event == "benchmark_system_prompt":
        return "System prompt (once per benchmark)"
    if event == "on_device_zero_calls":
        return "On-device returned 0 tool calls (raw response logged below)"
    return event


def _log_event(level: int, event: str, request_id: str | None = None, **kwargs: object) -> None:
    """Emit a single structured log line. Safe for production (no secrets)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    payload = {"event": event, "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if request_id is not None:
        payload["request_id"] = request_id
    for k, v in kwargs.items():
        if v is not None and k not in ("api_key", "password", "token"):
            payload[k] = v
    level_name = logging.getLevelName(level) if level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR) else "INFO"
    summary = _event_summary(event, payload)
    # Line: timestamp  LEVEL  [request_id]  →  Summary.  [optional k=v for grep]
    parts = [f"{ts}  {level_name:5}"]
    if request_id is not None:
        parts.append(f"[{request_id}]")
    parts.append("→")
    parts.append(summary)
    line = "  ".join(str(p) for p in parts)
    logger.log(level, line, extra={"payload": payload})


def log_event(level: int, event: str, request_id: str | None = None, **kwargs: object) -> None:
    """Public API for structured logging (e.g. from benchmark). Same contract as internal _log_event."""
    _log_event(level, event, request_id=request_id, **kwargs)


def _cactus_capability_context_from_file() -> str:
    """Build capability context string from capability_data.json. Returns empty string if file missing/invalid."""
    path = CHALLENGE_ROOT / "capability_data.json"
    if not path.exists():
        return ""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""
    if not data:
        return ""
    return "\n\n<capability_context>\n" + json.dumps(data, indent=2) + "\n</capability_context>"


def generate_cactus(messages, tools, request_id: str | None = None):
    """Run function calling on-device via FunctionGemma + Cactus. System prompt = prompts.STRICT_CACTUS_SYSTEM_PROMPT + capability context from capability_data.json."""
    rid = request_id or str(uuid.uuid4())[:8]
    _log_event(
        logging.INFO, "on_device_start", request_id=rid, num_tools=len(tools),
        input_messages=compact_messages_for_log(messages),
        input_tool_names=[t.get("name", "") for t in tools],
    )

    global _system_prompt_printed
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    system_content = STRICT_CACTUS_SYSTEM_PROMPT + _cactus_capability_context_from_file()
    if not _system_prompt_printed:
        _system_prompt_printed = True
        print("\n" + "=" * 60 + "\n[SYSTEM PROMPT — once per benchmark]\n" + "=" * 60 + "\n")
        print(system_content)
        print("=" * 60 + "\n")
        _log_event(
            logging.INFO, "benchmark_system_prompt", request_id=rid,
            system_prompt=system_content,
        )
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_content}] + messages,
        tools=cactus_tools,
        force_tools=True,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }

    # When on-device returns 0 tool calls, log what Cactus actually returned (so we can see format/parsing issues)
    if len(out["function_calls"]) == 0:
        _log_event(
            logging.INFO, "on_device_zero_calls", request_id=rid,
            raw_keys=list(raw.keys()),
            response_preview=(raw.get("response") or "")[:400] if raw.get("response") else None,
            raw_str_preview=raw_str[:500],
        )
    _log_event(
        logging.INFO, "on_device_complete", request_id=rid,
        confidence=out["confidence"], total_time_ms=out["total_time_ms"],
        num_function_calls=len(out["function_calls"]),
        function_calls=out["function_calls"],
    )
    return out


def generate_cloud(messages, tools, request_id: str | None = None):
    """Run function calling via Gemini Cloud API."""
    rid = request_id or str(uuid.uuid4())[:8]
    model_name = "gemini-2.5-flash"
    _log_event(
        logging.INFO, "cloud_start", request_id=rid, model=model_name, num_tools=len(tools),
        input_messages=compact_messages_for_log(messages),
        input_tool_names=[t.get("name", "") for t in tools],
    )

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]
    contents = [m["content"] for m in messages if m["role"] == "user"]
    start_time = time.time()

    gemini_response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )
    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    _log_event(
        logging.INFO, "cloud_complete", request_id=rid,
        total_time_ms=round(total_time_ms, 2), num_function_calls=len(function_calls),
        function_calls=function_calls,
    )
    return {"function_calls": function_calls, "total_time_ms": total_time_ms}




def generate_hybrid(messages, tools, confidence_threshold: float = 0.50, request_id: str | None = None):
    """Hybrid inference: use on-device when API confidence >= threshold and we got at least one tool call, else call cloud.
    Default 0.50: symmetric boundary (≥0.5 keep on-device, <0.5 fallback). Override with CONFIDENCE_THRESHOLD env.
    If on-device returns 0 tool calls we always fall back to cloud so we never surface empty on-device results."""
    rid = request_id or str(uuid.uuid4())[:8]
    local = generate_cactus(messages, tools, request_id=rid)

    has_calls = len(local.get("function_calls") or []) > 0
    if local["confidence"] >= confidence_threshold and has_calls:
        local["source"] = "on-device"
        _log_event(
            logging.INFO, "hybrid_decision", request_id=rid,
            source="on-device", local_confidence=local["confidence"],
            confidence_threshold=confidence_threshold, reason="confidence_above_threshold",
        )
        return local

    reason = "confidence_below_threshold" if local["confidence"] < confidence_threshold else "zero_tool_calls"
    _log_event(
        logging.INFO, "hybrid_decision", request_id=rid,
        source="cloud", local_confidence=local["confidence"],
        confidence_threshold=confidence_threshold, reason=reason,
    )
    try:
        cloud = generate_cloud(messages, tools, request_id=rid)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud
    except ClientError as e:
        _log_event(
            logging.WARNING, "cloud_fallback_failed", request_id=rid,
            error_type=type(e).__name__, status_code=getattr(e, "status_code", None),
            message=str(e)[:300], fallback="on-device",
        )
        local["source"] = "on-device (cloud skipped)"
        return local


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
