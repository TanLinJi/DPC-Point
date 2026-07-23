"""LLM-assisted prompt loading and generation for DPC-Point."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from utils.config import DEFAULT_PROMPT_CONFIG, PROJECT_ROOT, llm_env_config
from utils.prompt_utils import make_weighted_prompt_fusion

HANDCRAFTED_TEMPLATE_DIR = Path("text_templates/handcrafted")
ORIGINAL_HANDCRAFTED_TEMPLATE = "original_handcrafted"


def safe_name(name: str) -> str:
    name = str(name).replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def prompt_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    if key in {"modelnet", "modelnet_c", "modelnet-c"}:
        return "modelnet_c"
    if key in {"scanobjectnn", "scanobjectnn_c", "scanobjectnn-c", "sonn_c"}:
        return "sonn_c"
    return key


def strip_json_code_fence(text: str) -> str:
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def clean_prompt_text(text: str) -> str:
    text = str(text).strip().strip(",")
    text = text.replace('\\"', '"').strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text.strip().strip(",")


def parse_prompt_list(content: str) -> list[str]:
    content = strip_json_code_fence(content)
    if not content:
        return []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [clean_prompt_text(item) for item in data if clean_prompt_text(item)]
        if isinstance(data, dict):
            for key in ["prompts", "descriptions", "sentences", "items", "data"]:
                value = data.get(key)
                if isinstance(value, list):
                    return [clean_prompt_text(item) for item in value if clean_prompt_text(item)]
    except json.JSONDecodeError:
        pass

    left = content.find("[")
    right = content.rfind("]")
    if left != -1 and right != -1 and right > left:
        try:
            data = json.loads(content[left:right + 1])
            if isinstance(data, list):
                return [clean_prompt_text(item) for item in data if clean_prompt_text(item)]
        except json.JSONDecodeError:
            pass

    prompts = []
    for line in content.splitlines():
        line = re.sub(r"^[0-9]+[.)]\s*", "", line.strip())
        line = line.strip("- ").strip()
        if line and line not in {"[", "]", "{", "}"}:
            cleaned = clean_prompt_text(line)
            if cleaned:
                prompts.append(cleaned)
    return prompts


def read_prompt_json(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    prompts = data.get("prompts", data) if isinstance(data, dict) else data
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompt JSON must contain a class-to-prompts dictionary: {path}")
    return {str(key): [str(item) for item in value] for key, value in prompts.items()}


def load_handcrafted_templates(name: str, project_root: Path = PROJECT_ROOT) -> list[str]:
    template_path = project_root / HANDCRAFTED_TEMPLATE_DIR / f"{name}.json"
    if not template_path.exists():
        raise FileNotFoundError(f"Handcrafted template file not found: {template_path}")

    with template_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)

    templates = payload.get("templates") if isinstance(payload, dict) else payload
    if not isinstance(templates, list) or not all(isinstance(item, str) for item in templates):
        raise ValueError(f"Invalid handcrafted template format: {template_path}")
    return list(templates)


def default_prompt_filename(dataset_name: str, provider: str, model: str, prompt_mode: str, prompt_count: int) -> str:
    return "{}_{}_{}_{}_{}_prompts.json".format(
        safe_name(prompt_dataset_name(dataset_name)),
        safe_name(provider or "provider"),
        safe_name(model or "model"),
        safe_name(prompt_mode),
        int(prompt_count),
    )


def resolve_prompt_file(args, dataset_name: str, project_root: Path = PROJECT_ROOT) -> Path | None:
    cache_dir = Path(getattr(args, "prompt_cache_dir", DEFAULT_PROMPT_CONFIG.prompt_cache_dir))
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    explicit = getattr(args, "prompt_cache_file", "")
    if explicit:
        explicit_path = Path(explicit)
        if not explicit_path.is_absolute():
            explicit_path = cache_dir / explicit_path
        return explicit_path

    env = llm_env_config(project_root)
    provider = getattr(args, "provider", "") or env.get("provider", "")
    model = getattr(args, "model", "") or env.get("model", "")
    prompt_mode = getattr(args, "llm_prompt_mode", DEFAULT_PROMPT_CONFIG.llm_prompt_mode)
    prompt_count = int(getattr(args, "dynamic_prompt_count", DEFAULT_PROMPT_CONFIG.dynamic_prompt_count))
    exact = cache_dir / default_prompt_filename(dataset_name, provider, model, prompt_mode, prompt_count)
    if exact.exists():
        return exact

    dataset_prefix = safe_name(prompt_dataset_name(dataset_name))
    matches = sorted(cache_dir.glob(f"{dataset_prefix}_*_{safe_name(prompt_mode)}_{prompt_count}_prompts.json"))
    if matches:
        return matches[0]
    return exact


def build_llm_request(classname: str, prompt_count: int, model: str, temperature: float, prompt_mode: str) -> dict:
    if prompt_mode != "multiview_2d3d":
        raise ValueError(f"Only multiview_2d3d is enabled in the release generator, got {prompt_mode}.")
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate concise English class descriptions for vision-language recognition of 3D point clouds. "
                    "Return only a JSON array of strings. Do not include explanations, numbering, markdown, or extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate exactly {prompt_count} descriptions for the class '{classname}'. "
                    "Descriptions should include both 2D visual semantics and 3D point-cloud geometry. "
                    "Each description must be a complete English sentence. Return only a JSON array of strings."
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max(1200, int(prompt_count) * 110),
    }


def _chat_completion_url(base_url: str) -> str:
    base_url = str(base_url).strip().rstrip("/")
    if not base_url:
        raise RuntimeError("BASE_URL is empty in .env.")
    if base_url.endswith("chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def call_openai_compatible_api(api_key: str, base_url: str, payload: dict) -> str:
    request_data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _chat_completion_url(base_url),
        data=request_data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP error {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API request failed: {exc}") from exc
    response_json = json.loads(response_text)
    return response_json["choices"][0]["message"]["content"]


def generate_llm_prompts(classnames: list[str], args, dataset_name: str, project_root: Path = PROJECT_ROOT) -> dict[str, list[str]]:
    prompt_file = resolve_prompt_file(args, dataset_name, project_root)
    if prompt_file is not None and prompt_file.exists() and not getattr(args, "force_regenerate_prompts", False):
        return read_prompt_json(prompt_file)

    env = llm_env_config(project_root)
    api_key = env["api_key"]
    base_url = env["base_url"]
    model = getattr(args, "model", "") or env["model"]
    provider = getattr(args, "provider", "") or env["provider"]
    temperature = float(getattr(args, "temperature", env["temperature"]))
    prompt_count = int(getattr(args, "dynamic_prompt_count", DEFAULT_PROMPT_CONFIG.dynamic_prompt_count))
    prompt_mode = getattr(args, "llm_prompt_mode", DEFAULT_PROMPT_CONFIG.llm_prompt_mode)
    max_retries = int(getattr(args, "llm_max_retries", DEFAULT_PROMPT_CONFIG.llm_max_retries))

    if not api_key:
        raise RuntimeError("API_KEY is empty in .env, and no cached LLM prompt JSON was found.")
    if not model:
        raise RuntimeError("MODEL is empty in .env, and no cached LLM prompt JSON was found.")

    prompts = {}
    failed = []
    for classname in classnames:
        parsed = []
        for attempt in range(1, max_retries + 1):
            payload = build_llm_request(classname, prompt_count, model, temperature, prompt_mode)
            content = call_openai_compatible_api(api_key, base_url, payload)
            parsed = parse_prompt_list(content)
            if len(parsed) >= prompt_count:
                parsed = parsed[:prompt_count]
                break
            time.sleep(min(2 * attempt, 10))
        if len(parsed) < prompt_count:
            failed.append(classname)
        prompts[str(classname).replace("_", " ")] = parsed

    if failed:
        raise RuntimeError(f"LLM prompt generation failed for classes: {failed}")

    if prompt_file is not None:
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_source": "llm_descriptions",
            "llm_provider": provider,
            "llm_model": model,
            "llm_api_base_url": base_url,
            "llm_prompt_mode": prompt_mode,
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": prompt_dataset_name(dataset_name),
            "dynamic_prompt_count": prompt_count,
            "temperature": temperature,
            "class_names": list(classnames),
            "completed_class_names": list(prompts.keys()),
            "failed_classes": [],
            "prompts": prompts,
        }
        with prompt_file.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2, ensure_ascii=False)
    return prompts


def get_prompt_template(args, classnames: list[str], dataset_name: str, project_root: Path = PROJECT_ROOT):
    source = getattr(args, "prompt_source", DEFAULT_PROMPT_CONFIG.prompt_source)
    if source == "handcrafted":
        return load_handcrafted_templates(ORIGINAL_HANDCRAFTED_TEMPLATE, project_root)
    if source == "llm_descriptions":
        return generate_llm_prompts(classnames, args, dataset_name, project_root)
    if source == "handcrafted_with_llm":
        dynamic_template = generate_llm_prompts(classnames, args, dataset_name, project_root)
        return make_weighted_prompt_fusion(
            load_handcrafted_templates(ORIGINAL_HANDCRAFTED_TEMPLATE, project_root),
            dynamic_template,
            getattr(args, "prompt_static_weight", DEFAULT_PROMPT_CONFIG.prompt_static_weight),
            getattr(args, "prompt_dynamic_weight", DEFAULT_PROMPT_CONFIG.prompt_dynamic_weight),
        )
    raise ValueError(f"Unsupported prompt source: {source}")
