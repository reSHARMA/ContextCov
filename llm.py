"""Provider clients and model configuration for ContextCov.

Every stage of the pipeline resolves its model from an environment variable, so
you can use one model everywhere or a different one per stage.

Configuration
-------------
``DEFAULT_LLM`` sets the model for every stage, as ``provider:model[:temperature]``::

    DEFAULT_LLM=openai:gpt-4o:0.2

Any stage can override it with ``<STAGE>_LLM``, where ``<STAGE>`` is one of
COMPRESSION, ROUTER, STATIC_CHECK, PROCESS_CHECK, ARCH_DETERMINISTIC or
ARCH_SEMANTIC. A semicolon-separated list selects an ensemble.

Providers
---------
``openai``      OPENAI_API_KEY (OPENAI_BASE_URL optional, defaults to the public API)
``azure``       AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (AZURE_OPENAI_API_VERSION optional)
``gemini``      GEMINI_API_KEY, GEMINI_BASE_URL
``ollama``      OLLAMA_URL
``claudecode``  runs the local ``claude`` CLI instead of an HTTP API; the model
                string is passed to ``claude --model``. See CLAUDECODE_BIN,
                CLAUDECODE_EXTRA_ARGS, CLAUDECODE_TIMEOUT. Temperature is ignored.

Note
----
``.env`` is loaded from the directory containing this file rather than the
process working directory, so keys resolve when the tool is run from elsewhere.
"""

from __future__ import annotations

import json
import os
import random
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

# Project root (directory containing this file); .env is loaded from here, not from cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables from project .env if present (idempotent)
load_dotenv(_PROJECT_ROOT / ".env")

__all__ = [
    "get_client",
    "get_provider_and_model_from_env",
    "get_model_and_client",
    "build_multimodal_user_message",
    "run_multimodal_single",
    "chat_completion_create",
    "run_parallel_chat_completions",
    "is_claude_code_provider",
    "claude_code_completion",
    "ClaudeCodeClient",
]

# Optional lightweight response caching (see llm_cache.py). If the module is
# absent or errors, we silently proceed without caching.
try:  # pragma: no cover - optional
    from llm_cache import cached_chat_completion, get_cache
except Exception:  # noqa: pragma: no cover
    cached_chat_completion = None  # type: ignore
    get_cache = None  # type: ignore


_CLIENT_CACHE = {}


class ClaudeCodeClient:
    """Sentinel returned by get_client('claudecode'); not an OpenAI client."""




def is_claude_code_provider(provider: str) -> bool:
    """True if provider string selects Claude Code CLI (not an HTTP API)."""
    return (provider or "").strip().lower() == "claudecode"




















def _env_prepend_path(env: dict[str, str], prefix: str) -> None:
    """Prepend ``prefix`` to PATH (ContextCov shims)."""
    prefix = (prefix or "").strip()
    if not prefix:
        return
    prev = env.get("PATH", "") or ""
    env["PATH"] = f"{prefix}:{prev}" if prev else prefix




def _messages_to_claude_code_prompt(messages: List[Dict[str, Any]]) -> str:
    """Flatten chat messages into one prompt for `claude -p`."""
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip().upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_bits: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_bits.append(str(block.get("text") or ""))
                elif isinstance(block, dict) and "text" in block:
                    text_bits.append(str(block.get("text") or ""))
            content = "\n".join(text_bits)
        else:
            content = str(content)
        parts.append(f"### {role}\n{content.strip()}")
    return "\n\n".join(parts)


def claude_code_completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    cwd: str | None = None,
) -> str:
    """
    Run `claude -p` with the given model and prompt; return stdout text.

    Parameters
    ----------
    cwd
        Working directory for the CLI (typically the repository root). If None,
        inherits the current process directory.
    """
    # Ensure .env from this repo is merged into os.environ so the child `claude`
    # process inherits ANTHROPIC_API_KEY and related vars (cwd may be a clone).
    load_dotenv(_PROJECT_ROOT / ".env")
    prompt = _messages_to_claude_code_prompt(messages)
    bin_path = os.environ.get("CLAUDECODE_BIN", "claude").strip() or "claude"
    extra: List[str] = []
    extra_raw = os.environ.get("CLAUDECODE_EXTRA_ARGS", "").strip()
    if extra_raw:
        extra = shlex.split(extra_raw)
    try:
        timeout = int(os.environ.get("CLAUDECODE_TIMEOUT", "600"))
    except ValueError:
        timeout = 600
    # Pass the prompt on stdin so huge generator prompts do not hit ARG_MAX (E2BIG).
    cmd = [
        bin_path,
        "-p",
        "--model",
        model,
        "--dangerously-skip-permissions",
        *extra,
    ]
    r = subprocess.run(
        cmd,
        cwd=cwd if cwd else None,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout if timeout > 0 else None,
        env=os.environ,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(
            f"claude CLI exited {r.returncode}: {err[:4000] or '(no output)'}"
        )
    return (r.stdout or "").strip()




def _synthetic_chat_response_text(content: str) -> Any:
    """Minimal response object with .choices[0].message.content for generators."""

    class _Msg:
        def __init__(self, c: str) -> None:
            self.content = c
            self.role = "assistant"

    class _Choice:
        def __init__(self, c: str) -> None:
            self.message = _Msg(c)
            self.finish_reason = "stop"

    class _Resp:
        def __init__(self, c: str) -> None:
            self.choices = [_Choice(c)]
            self.usage = None
            self.model = None

    return _Resp(content)


def _get_openai_retry_config():
    """Return kwargs for client retry configuration if supported by SDK.

    Defaults to max_retries=1. Override with CLAUDE_PROXY_BACKEND_RETRIES or OPENAI_MAX_RETRIES.
    Provider gateways may still perform internal retries beyond SDK control.
    """
    retries = int(
        os.getenv(
            "CLAUDE_PROXY_BACKEND_RETRIES",
            os.getenv("OPENAI_MAX_RETRIES", "1"),
        )
    )
    if retries < 1:
        retries = 1
    return {"max_retries": retries}


def get_client(provider: str):
    """Return a cached client instance for the given provider.

    Parameters
    ----------
    provider : str
        One of: openai, azure, gemini, ollama, claudecode
    """
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    provider = provider.strip()

    if provider == "claudecode":
        c = ClaudeCodeClient()
        _CLIENT_CACHE[provider] = c
        return c



    if provider == "ollama":
        base_url = os.getenv("OLLAMA_URL")
        if not base_url:
            raise ValueError("OLLAMA_URL is not set")
        client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # placeholder; Ollama ignores key
        )
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        # OPENAI_BASE_URL is optional: default to the public API so a plain
        # OPENAI_API_KEY is enough. Use `or` rather than a getenv default so a
        # set-but-empty value also falls back instead of becoming base_url="".
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            **_get_openai_retry_config(),
        )
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is not set")
        if not os.getenv("GEMINI_BASE_URL"):
            raise ValueError("GEMINI_BASE_URL is not set")
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("GEMINI_BASE_URL"),
            **_get_openai_retry_config(),
        )
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            # Optional; matches the default used for the MS Azure provider above.
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            **_get_openai_retry_config(),
        )
        _CLIENT_CACHE[provider] = client
        return client


    raise ValueError(f"Invalid provider: {provider}")


def get_provider_and_model_from_env(
    prefix: str,
) -> Union[Tuple[str, str, float], Tuple[List[str], List[str], List[float]]]:
    """Resolve provider/model(/temperature) triple(s) from environment.

    Order of lookup: <PREFIX>_LLM then DEFAULT_LLM.
    Accepts single: provider:model[:temp]
    Or list: provider:model[:temp];provider2:model2[:temp];...
    Temperature defaults to 1.0 if omitted.
    Returns either a single (provider, model, temperature) OR three lists.
    """
    env_var = f"{prefix.upper()}_LLM"
    value = os.getenv(env_var) or os.getenv("DEFAULT_LLM")
    if not value:
        # prefix == "DEFAULT" would otherwise render "Neither DEFAULT_LLM nor DEFAULT_LLM".
        if env_var == "DEFAULT_LLM":
            raise ValueError("DEFAULT_LLM is not set in environment.")
        raise ValueError(f"Neither {env_var} nor DEFAULT_LLM is set in environment.")

    # Multi-entry form
    if ";" in value:
        providers: List[str] = []
        models: List[str] = []
        temperatures: List[float] = []
        for pair in value.split(";"):
            parts = pair.split(":")
            if len(parts) < 2:
                raise ValueError(f"Invalid provider:model pair: {pair}")
            provider = parts[0].strip()
            temperature = 1.0
            model_parts = parts[1:]
            if len(parts) > 2:
                try:
                    temperature = float(parts[-1].strip())
                    model_parts = parts[1:-1]
                except ValueError:
                    model_parts = parts[1:]
            model = ":".join(model_parts).strip()
            providers.append(provider)
            models.append(model)
            temperatures.append(temperature)
        return providers, models, temperatures

    # Single entry
    parts = value.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid provider:model format in {env_var} or DEFAULT_LLM: {value}"
        )
    provider = parts[0].strip()
    temperature = 1.0
    model_parts = parts[1:]
    if len(parts) > 2:
        try:
            temperature = float(parts[-1].strip())
            model_parts = parts[1:-1]
        except ValueError:
            model_parts = parts[1:]
    model = ":".join(model_parts).strip()
    return provider, model, temperature


def get_model_and_client(prefix: str):
    """Convenience helper returning (client, model, temperature).

    Example:
        client, model, temp = get_model_and_client("spec_checker")
    """
    result = get_provider_and_model_from_env(prefix)
    if isinstance(result[0], list):  # type: ignore[index]
        providers, models, temps = result  # type: ignore[assignment]
        clients = [get_client(p) for p in providers]
        return clients, models, temps
    provider, model, temp = result  # type: ignore[misc]
    client = get_client(provider)
    return client, model, temp

# ------------------------------------------------------------
# Image + LLM Diff Support (stub integration layer)
# ------------------------------------------------------------
import base64
from typing import Optional, Dict, Any

def _encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def build_multimodal_user_message(text: str, image_path: str) -> Dict[str, Any]:
    """Return a message dict suitable for OpenAI / Azure-style multimodal chat."""
    import base64
    import mimetypes
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image for multimodal message not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode('utf-8')
    data_url = f"data:{mime};base64,{b64}"
    content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    return {"role": "user", "content": content}

def _needs_responses_api(model: str, provider: str) -> bool:
    """Determine if a model requires the Responses API instead of Chat Completions API.
    
    The Responses API is used for newer models like GPT-5.2. This function detects
    models that require the Responses API based on model name patterns.
    
    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5.2-chat", "gpt-5.2")
    provider : str
        Provider name (e.g., "openai", "azure")
    
    Returns
    -------
    bool
        True if the model requires Responses API, False otherwise
    """
    # Models that require Responses API (typically GPT-5.x series)
    responses_api_models = [
        "gpt-5",
        "gpt-5.2",
    ]
    
    model_lower = model.lower()
    for pattern in responses_api_models:
        if model_lower.startswith(pattern):
            return True
    
    return False


def _convert_messages_to_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Chat Completions messages format to Responses API input format.
    
    The Responses API uses a different content structure:
    - User messages: {"role": "user", "content": [{"type": "input_text", "text": "..."}, {"type": "input_image", "image_url": "..."}]}
    - Assistant messages: {"role": "assistant", "content": [{"type": "output_text", "text": "..."}]}
    
    Key differences:
    - User messages: "text" type becomes "input_text", "image_url" becomes "input_image"
    - Assistant messages: "text" type becomes "output_text" (not "input_text"!)
    
    Parameters
    ----------
    messages : List[Dict[str, Any]]
        Messages in Chat Completions format
    
    Returns
    -------
    List[Dict[str, Any]]
        Messages in Responses API input format
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Determine the correct text type based on role
        # User messages use "input_text", assistant messages use "output_text"
        text_type = "output_text" if role == "assistant" else "input_text"
        
        # Handle different content formats
        if isinstance(content, str):
            # Simple string content
            converted_content = [{"type": text_type, "text": content}]
        elif isinstance(content, list):
            # Already a list, convert each item
            converted_content = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "text")
                    # Convert "text" type to the appropriate type based on role
                    if item_type == "text":
                        converted_content.append({
                            "type": text_type,
                            "text": item.get("text", "")
                        })
                    elif item_type == "image_url":
                        # Convert "image_url" to "input_image" for Responses API
                        # Note: images are only valid for user messages, not assistant
                        if role == "user":
                            # Chat Completions format: {"type": "image_url", "image_url": {"url": "...", "detail": "high"}}
                            # Responses API format: {"type": "input_image", "image_url": "..."}
                            image_url_obj = item.get("image_url", {})
                            if isinstance(image_url_obj, dict):
                                # Extract the URL from the nested structure
                                image_url = image_url_obj.get("url", "")
                            else:
                                # If it's already a string
                                image_url = str(image_url_obj)
                            
                            converted_content.append({
                                "type": "input_image",
                                "image_url": image_url
                            })
                        # Skip images in assistant messages (not supported)
                    elif item_type in ["output_text", "input_text", "input_image"]:
                        # Already in Responses API format, keep as-is
                        converted_content.append(item)
                    else:
                        # Keep other types as-is
                        converted_content.append(item)
                else:
                    # If it's not a dict, wrap it with the appropriate type
                    converted_content.append({"type": text_type, "text": str(item)})
        else:
            # Fallback: convert to string
            converted_content = [{"type": text_type, "text": str(content)}]
        
        converted.append({
            "role": role,
            "content": converted_content
        })
    
    return converted


def _normalize_response(resp: Any, use_responses_api: bool) -> Dict[str, Any]:
    """Normalize response from either API format to a common structure.
    
    Parameters
    ----------
    resp : Any
        Response object from either API
    use_responses_api : bool
        Whether the response came from Responses API
    
    Returns
    -------
    Dict[str, Any]
        Normalized response with 'choices' and 'message' structure
    """
    if use_responses_api:
        # Responses API format - extract from response structure
        # According to Responses API docs: https://platform.openai.com/docs/guides/migrate-to-responses
        # The response has an 'output' field, and the actual text content is in output.text
        # However, output.text might be a ResponseTextConfig object, not the actual text
        # The actual text content should be extracted from the response structure
        try:
            content = None
            
            # Method 1: Try SDK convenience property first (Python SDK has output_text)
            # According to official docs: "SDK-only convenience property that contains 
            # the aggregated text output from all output_text items in the output array"
            if content is None and hasattr(resp, 'output_text'):
                output_text = resp.output_text
                if isinstance(output_text, str) and output_text:
                    content = output_text
            
            # Method 2: Try to access output directly (official structure)
            # According to Responses API docs: 
            # response.output[0] is a message object with type="message" and role="assistant"
            # response.output[0].content[0] is an object with type="output_text" and text="actual text"
            if content is None and hasattr(resp, 'output'):
                try:
                    output = resp.output
                    # output is a list
                    if output and isinstance(output, list) and len(output) > 0:
                        # Get first output message (should be type="message", role="assistant")
                        output_msg = output[0]
                        # Check if it has content attribute (array of content items)
                        if hasattr(output_msg, 'content'):
                            content_list = output_msg.content
                            if content_list and isinstance(content_list, list) and len(content_list) > 0:
                                # Get first content item (should be type="output_text")
                                content_item = content_list[0]
                                # Extract text from ResponseOutputText
                                if hasattr(content_item, 'text'):
                                    text_val = content_item.text
                                    if isinstance(text_val, str) and text_val:
                                        content = text_val
                                    elif text_val is not None:
                                        # Convert to string if it's not None and not empty
                                        content = str(text_val) if str(text_val).strip() else None
                                elif isinstance(content_item, dict):
                                    # If it's a dict, get the 'text' field
                                    content = content_item.get('text')
                                elif hasattr(content_item, 'content'):
                                    content = content_item.content
                except Exception as e:
                    # If direct access fails, fall through to other methods
                    pass
            
            # Method 2: Use model_dump to get the full structure and extract text (fallback)
            if content is None and hasattr(resp, 'model_dump'):
                try:
                    dump = resp.model_dump()
                    if 'output' in dump:
                        output_dump = dump['output']
                        # output is a list of ResponseOutputMessage objects
                        if isinstance(output_dump, list) and len(output_dump) > 0:
                            # Get first output message
                            output_msg_dump = output_dump[0]
                            if isinstance(output_msg_dump, dict):
                                # Check for content field (list of ResponseOutputText objects)
                                if 'content' in output_msg_dump:
                                    content_list = output_msg_dump['content']
                                    if isinstance(content_list, list) and len(content_list) > 0:
                                        # Get first content item
                                        content_item = content_list[0]
                                        if isinstance(content_item, dict):
                                            # Extract text from ResponseOutputText
                                            content = content_item.get('text')
                                        elif hasattr(content_item, 'text'):
                                            content = content_item.text
                                # Alternative: check for text directly in output message
                                if content is None and 'text' in output_msg_dump:
                                    content = output_msg_dump['text']
                except Exception as e:
                    pass
            
            # Method 3: Check for text attribute directly on response
            if content is None and hasattr(resp, 'text'):
                text_val = resp.text
                if isinstance(text_val, str):
                    content = text_val
            
            # If none of the documented accessors produced text, fail loudly.
            # The previous fallback scraped the SDK's repr() with a regex, which
            # silently produced truncated or wrong content when the SDK changed.
            if content is None:
                raise RuntimeError(
                    "Could not extract text from the Responses API result "
                    f"(type={type(resp).__name__}). The SDK response shape may have changed."
                )
            
            # Get finish_reason if available
            finish_reason = 'stop'
            if hasattr(resp, 'finish_reason'):
                finish_reason = resp.finish_reason
            elif hasattr(resp, 'model_dump'):
                try:
                    dump = resp.model_dump()
                    if 'finish_reason' in dump:
                        finish_reason = dump['finish_reason']
                except Exception:
                    pass
            
            # Create a normalized response structure
            return {
                "choices": [{
                    "message": {
                        "content": content,
                        "role": "assistant"
                    },
                    "finish_reason": finish_reason
                }],
                "usage": getattr(resp, 'usage', None),
                "model": getattr(resp, 'model', None),
                "raw": resp
            }
        except Exception as e:
            # Fallback: return raw response with error info
            return {
                "choices": [{
                    "message": {
                        "content": str(resp),
                        "role": "assistant"
                    },
                    "finish_reason": "unknown"
                }],
                "raw": resp,
                "error": str(e)
            }
    else:
        # Chat Completions API format - already in expected structure
        return {
            "choices": resp.choices if hasattr(resp, 'choices') else [],
            "usage": getattr(resp, 'usage', None),
            "model": getattr(resp, 'model', None),
            "raw": resp
        }


def chat_completion_create(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    provider: str = "openai",
    temperature: float = 1.0,
    claude_code_cwd: str | None = None,
    **kwargs
) -> Any:
    """Unified interface for creating chat completions that supports both APIs.
    
    This function automatically detects whether to use the Chat Completions API
    (messages format) or the Responses API (input format) based on the model name.
    When LLM cache is enabled (e.g. LLM_CACHE_ENABLE=1), lookups and stores are
    done here so all callers (including run_semantic_checks) get caching.
    
    Parameters
    ----------
    client : Any
        OpenAI or AzureOpenAI client instance
    model : str
        Model name (e.g., "gpt-4o", "gpt-5.2-chat")
    messages : List[Dict[str, Any]]
        Messages in Chat Completions format (will be converted if needed)
    provider : str
        Provider name (e.g., "openai", "azure") - used for detection
    temperature : float
        Temperature parameter
    **kwargs
        Additional parameters to pass to the API call
    
    Returns
    -------
    Any
        Response object (normalized to have .choices[0].message.content structure)
    """
    if is_claude_code_provider(provider):
        cwd = claude_code_cwd
        if not cwd:
            cwd = os.getcwd()
        text = claude_code_completion(model, messages, cwd=cwd)
        return _synthetic_chat_response_text(text)

    # Cache lookup when enabled (covers run_semantic_checks and any direct caller)
    meta = {"provider": provider, **kwargs}
    if _llm_cache_enabled() and get_cache is not None:
        cache = get_cache()
        key = cache.build_key(model, temperature, messages, meta)
        cached = cache.get(key)
        if cached is not None:
            return _response_from_cached_dict(cached)

    use_responses_api = _needs_responses_api(model, provider)
    
    if use_responses_api:
        # Use Responses API
        input_messages = _convert_messages_to_input(messages)
        
        # Build Responses API parameters
        responses_params = {
            "model": model,
            "input": input_messages,
            "text": {
                "format": {
                    "type": "text"
                }
            },
        }
        
        # Note: Responses API (e.g. GPT-5.2) does not support temperature or response_format.
        # Filter out Chat Completions–only kwargs to avoid Errors API errors.
        excluded = {"temperature", "response_format"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in excluded}
        responses_params.update(filtered_kwargs)
        
        # Call Responses API
        resp = client.responses.create(**responses_params)
        
        # Normalize response to match Chat Completions structure
        normalized = _normalize_response(resp, use_responses_api=True)
        
        # Create response-like objects with attribute access
        class Message:
            """Message object with attribute access."""
            def __init__(self, data):
                if isinstance(data, dict):
                    content = data.get("content")
                    # Ensure content is a string
                    if content is not None and not isinstance(content, str):
                        # If content is an object, try to extract string from it
                        if hasattr(content, 'text'):
                            content = content.text
                        elif hasattr(content, 'content'):
                            content = content.content
                        elif hasattr(content, 'value'):
                            content = content.value
                        elif not isinstance(content, str):
                            # Last resort: convert to string, but filter out config objects
                            content_str = str(content)
                            # If it looks like a Response object or config object, try to extract text
                            if 'Response(' in content_str and 'output=' in content_str:
                                # This is a Response object string representation - try to extract text
                                # The documented accessors did not yield text.
                                # Previously this scraped the SDK's repr() with a
                                # regex, which silently produced truncated content.
                                content = None
                            elif 'ResponseTextConfig' in content_str or 'format=' in content_str:
                                content = None
                            else:
                                content = content_str
                    self.content = content
                    self.role = data.get("role", "assistant")
                else:
                    content = getattr(data, 'content', None) if hasattr(data, 'content') else data
                    # Ensure content is a string
                    if content is not None and not isinstance(content, str):
                        if hasattr(content, 'text'):
                            content = content.text
                        elif hasattr(content, 'content'):
                            content = content.content
                        elif hasattr(content, 'value'):
                            content = content.value
                        elif not isinstance(content, str):
                            content_str = str(content)
                            if 'Response(' in content_str and 'output=' in content_str:
                                # This is a Response object - try to extract text
                                # The documented accessors did not yield text.
                                # Previously this scraped the SDK's repr() with a
                                # regex, which silently produced truncated content.
                                content = None
                            elif 'ResponseTextConfig' in content_str or 'format=' in content_str:
                                content = None
                            else:
                                content = content_str
                    self.content = content
                    self.role = getattr(data, 'role', 'assistant') if hasattr(data, 'role') else 'assistant'
            
            def __getattr__(self, name):
                return None
        
        class Choice:
            """Choice object with attribute access."""
            def __init__(self, data):
                if isinstance(data, dict):
                    self.message = Message(data.get("message", {}))
                    self.finish_reason = data.get("finish_reason", "stop")
                else:
                    self.message = getattr(data, 'message', None) if hasattr(data, 'message') else Message({})
                    self.finish_reason = getattr(data, 'finish_reason', 'stop') if hasattr(data, 'finish_reason') else 'stop'
            
            def __getattr__(self, name):
                return None
        
        class NormalizedResponse:
            """Response object that mimics Chat Completions API structure."""
            def __init__(self, data):
                # Convert choices from dicts to Choice objects
                choices_data = data.get("choices", [])
                self.choices = [Choice(choice) if not isinstance(choice, Choice) else choice for choice in choices_data]
                self.usage = data.get("usage")
                self.model = data.get("model")
                self._raw = data.get("raw")
                self._normalized_data = data
            
            def __getattr__(self, name):
                # Fallback to raw response if attribute not found
                if self._raw is not None and hasattr(self._raw, name):
                    return getattr(self._raw, name)
                # Also check normalized_data
                if name in self._normalized_data:
                    return self._normalized_data[name]
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def model_dump(self):
                """Return a dict representation of the response."""
                return {
                    "choices": [
                        {
                            "message": {
                                "content": choice.message.content if hasattr(choice, 'message') else None,
                                "role": choice.message.role if hasattr(choice, 'message') else "assistant"
                            },
                            "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else "stop"
                        }
                        for choice in self.choices
                    ],
                    "usage": self.usage,
                    "model": self.model
                }
        # Store successful result in cache when enabled (e.g. for run_semantic_checks)
        if _llm_cache_enabled() and get_cache is not None and normalized.get("choices"):
            try:
                cache = get_cache()
                key = cache.build_key(model, temperature, messages, meta)
                to_store = {k: v for k, v in normalized.items() if k != "raw"}
                cache.set(key, {"model": model, "temperature": temperature, "messages": messages, "meta": meta}, to_store)
            except Exception:
                pass
        return NormalizedResponse(normalized)
    else:
        # Use Chat Completions API (standard format)
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        params.update(kwargs)
        
        resp = client.chat.completions.create(**params)
        if _llm_cache_enabled() and get_cache is not None:
            serialized = _serialize_response(resp)
            if serialized and serialized.get("choices"):
                try:
                    cache = get_cache()
                    key = cache.build_key(model, temperature, messages, meta)
                    cache.set(key, {"model": model, "temperature": temperature, "messages": messages, "meta": meta}, serialized)
                except Exception as e:
                    # Warn rather than swallow: a silently dead cache means every
                    # rerun pays full API cost while appearing to be cached.
                    print(f"[llm] warning: could not write to LLM cache: {e}", file=sys.stderr)
        return resp


def _llm_cache_enabled() -> bool:
    """True if LLM cache is available and enabled (e.g. via LLM_CACHE_ENABLE=1)."""
    if get_cache is None:
        return False
    try:
        return not get_cache().disabled
    except Exception:
        return False


def _serialize_response(resp: Any) -> Optional[Dict[str, Any]]:
    """Serialize a chat completion response to a JSON-serializable dict for cache storage."""
    try:
        choices = getattr(resp, "choices", None)
        if not choices:
            return None
        out_choices = []
        for c in choices:
            msg = getattr(c, "message", None)
            if msg is None:
                continue
            content = getattr(msg, "content", None)
            role = getattr(msg, "role", "assistant")
            finish = getattr(c, "finish_reason", "stop")
            out_choices.append({
                "message": {"content": content, "role": role},
                "finish_reason": finish,
            })
        if not out_choices:
            return None
        return {
            "choices": out_choices,
            # The SDK's usage object is a pydantic model and is not JSON
            # serializable; storing it directly made every cache write raise.
            "usage": _usage_to_dict(getattr(resp, "usage", None)),
            "model": getattr(resp, "model", None),
        }
    except Exception:
        return None


def _usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """Convert an SDK usage object to a plain dict, or None if it cannot be."""
    if usage is None or isinstance(usage, dict):
        return usage
    for attr in ("model_dump", "dict"):
        fn = getattr(usage, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return {
        k: getattr(usage, k)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(getattr(usage, k, None), int)
    } or None


def _response_from_cached_dict(data: Dict[str, Any]) -> Any:
    """Build a response-like object from a cached dict (has .choices[0].message.content)."""
    class _Msg:
        def __init__(self, d: dict):
            self.content = d.get("content")
            self.role = d.get("role", "assistant")
    class _Choice:
        def __init__(self, d: dict):
            self.message = _Msg(d.get("message", {}))
            self.finish_reason = d.get("finish_reason", "stop")
    class _CachedResponse:
        def __init__(self, d: dict):
            self.choices = [_Choice(c) for c in d.get("choices", [])]
            self.usage = d.get("usage")
            self.model = d.get("model")
    return _CachedResponse(data)


def _is_rate_limit_error(ex: BaseException) -> bool:
    """True if the exception indicates a rate limit (429) or capacity error."""
    if getattr(ex, "status_code", None) == 429:
        return True
    name = type(ex).__name__
    if name == "RateLimitError":
        return True
    msg = str(ex).lower()
    return (
        "rate limit" in msg
        or "429" in msg
        or "too many requests" in msg
        or "no_capacity" in msg
        or "high demand" in msg
    )


def _is_no_capacity_error(ex: BaseException) -> bool:
    """True if the error is server overload / no_capacity (needs longer backoff)."""
    msg = str(ex).lower()
    return "no_capacity" in msg or "high demand" in msg or "peak load" in msg


def _chat_one_with_retry(
    request: Dict[str, Any],
    max_retries: int = 15,
    initial_backoff: float = 1.0,
) -> Any:
    """
    Call chat_completion_create once with retries on rate limit / capacity errors.
    Caching is handled inside chat_completion_create so all callers (including
    run_semantic_checks) get cache lookup/store. Waits with exponential backoff
    (plus jitter) on rate limit; uses longer backoff for no_capacity / high demand.
    """
    reserved = {"client", "model", "messages", "provider", "temperature", "claude_code_cwd"}
    extra = {k: v for k, v in request.items() if k not in reserved and not (k.startswith("_"))}
    backoff = initial_backoff
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return chat_completion_create(
                client=request["client"],
                model=request["model"],
                messages=request["messages"],
                provider=request.get("provider", "openai"),
                temperature=request.get("temperature", 1.0),
                claude_code_cwd=request.get("claude_code_cwd"),
                **extra,
            )
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e) and attempt < max_retries:
                # No-capacity / high demand: use longer backoff so server can recover
                if _is_no_capacity_error(e):
                    wait = min(backoff * 2.0, 300.0)  # cap 5 min for capacity errors
                    if attempt == 0:
                        wait = max(wait, 30.0)  # first no_capacity: wait at least 30s
                else:
                    wait = min(backoff * 2.0, 60.0)  # cap 60s for normal rate limit
                jitter = random.uniform(0, min(5.0, wait * 0.2))  # avoid thundering herd
                time.sleep(wait + jitter)
                backoff = wait
                continue
            raise
    raise last_exc  # type: ignore[misc]


def run_parallel_chat_completions(
    requests: List[Dict[str, Any]],
    max_concurrency: int = 5,
    max_retries: int = 15,
    initial_backoff: float = 2.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    """
    Run multiple chat completion requests in parallel with a bounded pool.

    Maintains a buffer of up to max_concurrency in-flight requests. On rate limit
    (429) or no_capacity, each worker retries with exponential backoff and jitter.
    no_capacity uses longer waits (30s–5 min). Results are returned in the same
    order as requests.

    Parameters
    ----------
    requests : List[Dict[str, Any]]
        Each dict must have client, model, messages; may have provider, temperature, **kwargs.
    max_concurrency : int
        Max number of requests in flight at once (default 5). Tune per model rate limits.
    max_retries : int
        Retries per request on rate limit / capacity (default 15).
    initial_backoff : float
        Initial backoff seconds (default 2.0), doubled on each retry (cap 60s or 300s for no_capacity).
    progress_callback : Optional[Callable[[int, int], None]]
        If set, called as progress_callback(done_count, total) as results complete.

    Returns
    -------
    List[Any]
        Responses in same order as requests (each has .choices[0].message.content).
    """
    total = len(requests)
    if total == 0:
        return []
    results: List[Any] = [None] * total  # type: ignore[list-item]
    done = 0

    def do_one(index: int) -> Tuple[int, Any]:
        resp = _chat_one_with_retry(
            requests[index],
            max_retries=max_retries,
            initial_backoff=initial_backoff,
        )
        return (index, resp)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(do_one, i): i for i in range(total)}
        for future in as_completed(futures):
            index, resp = future.result()
            results[index] = resp
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)
    return results


def run_multimodal_single(user_message: Dict[str, Any], prefix: str = "DEFAULT") -> Dict[str, Any]:
    """Execute a single-turn multimodal chat if provider supports images.

    Integrates optional caching (llm_cache.cached_chat_completion) keyed on
    (model, temperature, message content, prefix). Caching can be disabled via
    environment variables (see llm_cache.py) or if llm_cache import fails.
    """
    try:
        client, model, temp = get_model_and_client(prefix)
        # Get provider for API detection
        provider_result = get_provider_and_model_from_env(prefix)
        if isinstance(provider_result[0], list):
            provider = provider_result[0][0]  # type: ignore
        else:
            provider = provider_result[0]  # type: ignore
    except Exception as e:  # noqa
        return {"success": False, "error": f"Client init failed: {e}"}

    messages = [user_message]

    def _invoke():
        try:
            resp = chat_completion_create(
                client=client,
                model=model,
                messages=messages,
                provider=provider,
                temperature=temp,
            )
            content = resp.choices[0].message.content if resp.choices else None
            raw_repr = getattr(resp, 'model_dump', lambda: str(resp))()
            return {"success": True, "response": content, "raw": raw_repr}
        except Exception as e:  # noqa
            return {"success": False, "error": f"Invocation failed: {e}"}

    if cached_chat_completion is None:
        return _invoke()

    # Use optional meta to distinguish by prefix.
    result = cached_chat_completion(
        _invoke,
        model=model,
        temperature=temp,
        messages=messages,
        meta={"prefix": prefix, "multimodal": True},
    )
    return result