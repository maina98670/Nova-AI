"""Nova CodeBench: a Hugging Face powered coding-model evaluation app."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import requests


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
TASKS_PATH = DATA_DIR / "tasks.json"
HISTORY_PATH = DATA_DIR / "history.json"

CATEGORIES = [
    "App creation",
    "Bug fixing",
    "Code review",
    "Testing",
    "Web frontend",
    "Mobile app",
    "Backend/API",
    "Database",
    "DevOps",
    "Other",
]

STARTER_TASKS: list[dict[str, Any]] = [
    {
        "id": "starter-python-bug",
        "title": "Fix a Python bug and add tests",
        "category": "Bug fixing",
        "prompt": (
            "Fix the function below so it returns True only when a list contains "
            "at least one number and every item is a finite number. Return the "
            "corrected Python function and pytest tests for empty input, strings, "
            "NaN, infinity, integers, and floats.\n\n"
            "```python\n"
            "def valid_measurements(values):\n"
            "    return all(isinstance(value, (int, float)) for value in values)\n"
            "```"
        ),
        "rubric": (
            "Rejects an empty list, non-numbers, NaN, and positive or negative "
            "infinity; accepts finite integers and floats; includes useful tests."
        ),
    },
    {
        "id": "starter-web-app",
        "title": "Plan and build a small web app",
        "category": "Web frontend",
        "prompt": (
            "Create a small responsive task-list app using only HTML, CSS, and "
            "vanilla JavaScript. It must let a user add a task, mark it complete, "
            "delete it, and keep the list after refreshing the page. Return the "
            "complete contents of index.html, styles.css, and app.js, with each "
            "file in its own code block."
        ),
        "rubric": (
            "All three files are complete and connected correctly; add, complete, "
            "delete, and local persistence work; layout is usable on a phone."
        ),
    },
    {
        "id": "starter-api",
        "title": "Design a tested REST API",
        "category": "Backend/API",
        "prompt": (
            "Write a minimal FastAPI service for managing books. Include a Book "
            "schema, GET /books, GET /books/{book_id}, POST /books, and DELETE "
            "/books/{book_id}. Use in-memory storage, validate required fields, "
            "return appropriate HTTP errors, and provide pytest tests. Show the "
            "complete source files and commands to run the service and tests."
        ),
        "rubric": (
            "Routes and schemas are coherent; missing IDs and invalid input are "
            "handled; tests cover success and failure cases; run commands match files."
        ),
    },
    {
        "id": "starter-sql-review",
        "title": "Review a database migration",
        "category": "Database",
        "prompt": (
            "Review this SQL table for a small inventory app and return a safer "
            "PostgreSQL migration plus a short explanation. Preserve the intended "
            "fields while addressing data integrity and query performance.\n\n"
            "```sql\n"
            "CREATE TABLE stock (id text, product_name text, quantity text, "
            "updated_at text);\n"
            "```"
        ),
        "rubric": (
            "Uses appropriate types, primary key and not-null constraints, "
            "nonnegative quantity validation, timestamp type/default, and a "
            "reasonable index only when justified."
        ),
    },
]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, value: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def get_tasks() -> list[dict[str, Any]]:
    tasks = _read_json(TASKS_PATH, None)
    if not isinstance(tasks, list):
        tasks = list(STARTER_TASKS)
        _write_json(TASKS_PATH, tasks)
    return tasks


def get_history() -> list[dict[str, Any]]:
    history = _read_json(HISTORY_PATH, [])
    return history if isinstance(history, list) else []


def _task_choices() -> list[tuple[str, str]]:
    return [(task.get("title", "Untitled task"), task["id"]) for task in get_tasks()]


def load_task(task_id: str | None) -> tuple[str, str, str, str]:
    if not task_id:
        return "", CATEGORIES[0], "", ""
    task = next((item for item in get_tasks() if item.get("id") == task_id), None)
    if task is None:
        return "", CATEGORIES[0], "", ""
    return (
        task.get("title", ""),
        task.get("category", CATEGORIES[0]),
        task.get("prompt", ""),
        task.get("rubric", ""),
    )


def save_task(title: str, category: str, prompt: str, rubric: str) -> tuple[Any, str]:
    title = (title or "").strip()
    prompt = (prompt or "").strip()
    if not title or not prompt:
        return gr.Dropdown(choices=_task_choices()), "Enter a task title and prompt first."

    tasks = get_tasks()
    task_id = str(uuid.uuid4())
    for existing in tasks:
        if existing.get("title", "").strip().casefold() == title.casefold():
            task_id = existing["id"]
            break
    task = {
        "id": task_id,
        "title": title,
        "category": category if category in CATEGORIES else "Other",
        "prompt": prompt,
        "rubric": (rubric or "").strip(),
    }
    tasks = [item for item in tasks if item.get("id") != task_id]
    tasks.insert(0, task)
    _write_json(TASKS_PATH, tasks)
    return gr.Dropdown(choices=_task_choices(), value=task_id), "Task saved to this app's local task library."


def _chat_url(mode: str, endpoint_url: str) -> str:
    if mode == "Hugging Face Inference Providers":
        return "https://router.huggingface.co/v1/chat/completions"

    url = (endpoint_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Enter your Hugging Face Inference Endpoint URL.")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _call_model(
    mode: str,
    model_id: str,
    endpoint_url: str,
    token: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("Enter the Hugging Face model ID or served model name.")

    token = (token or "").strip() or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("Enter an HF token here or set HF_TOKEN in the app's environment/secrets.")

    messages: list[dict[str, str]] = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            _chat_url(mode, endpoint_url),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=(20, 240),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the model endpoint: {exc}") from exc

    elapsed = time.perf_counter() - started
    if not response.ok:
        detail = ""
        try:
            body = response.json()
            detail = body.get("error", body)
        except ValueError:
            detail = response.text[:1000]
        raise RuntimeError(f"Hugging Face returned HTTP {response.status_code}: {detail}")

    try:
        body = response.json()
        answer = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The endpoint responded, but the response was not a chat-completion response. Check the endpoint API format.") from exc

    if isinstance(answer, list):
        answer = "\n".join(part.get("text", "") for part in answer if isinstance(part, dict))
    usage = (body.get("usage") or {}) if isinstance(body, dict) else {}
    return {
        "answer": str(answer),
        "seconds": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def _record_result(
    title: str,
    category: str,
    prompt: str,
    rubric: str,
    model_label: str,
    mode: str,
    model_id: str,
    result: dict[str, Any],
) -> str:
    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_title": title.strip() or "Untitled evaluation",
        "category": category,
        "prompt": prompt,
        "rubric": rubric,
        "model_label": model_label,
        "mode": mode,
        "model_id": model_id,
        "response": result["answer"],
        "seconds": result["seconds"],
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "ratings": None,
    }
    history = get_history()
    history.insert(0, record)
    _write_json(HISTORY_PATH, history)
    return record_id


def run_evaluation(
    title: str,
    category: str,
    prompt: str,
    rubric: str,
    system_prompt: str,
    mode_a: str,
    model_id_a: str,
    endpoint_a: str,
    token_a: str,
    compare: bool,
    mode_b: str,
    model_id_b: str,
    endpoint_b: str,
    token_b: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str, str, str, str, str | None, str | None]:
    if not (prompt or "").strip():
        return "Add a coding task before running an evaluation.", "", "", "", "", None, None

    try:
        result_a = _call_model(mode_a, model_id_a, endpoint_a, token_a, system_prompt, prompt, temperature, max_tokens)
        id_a = _record_result(title, category, prompt, rubric, "Model A", mode_a, model_id_a, result_a)
        meta_a = f"**{model_id_a}** · {result_a['seconds']} sec · input tokens: {result_a.get('prompt_tokens', 'n/a')} · output tokens: {result_a.get('completion_tokens', 'n/a')}"
    except Exception as exc:  # Keep connection errors visible in the app instead of losing the task text.
        return f"Model A failed: {exc}", "", "", "", "", None, None

    if not compare:
        return "Model A response saved. Rate it below or open History.", result_a["answer"], meta_a, "", "", id_a, None

    try:
        result_b = _call_model(mode_b, model_id_b, endpoint_b, token_b or token_a, system_prompt, prompt, temperature, max_tokens)
        id_b = _record_result(title, category, prompt, rubric, "Model B", mode_b, model_id_b, result_b)
        meta_b = f"**{model_id_b}** · {result_b['seconds']} sec · input tokens: {result_b.get('prompt_tokens', 'n/a')} · output tokens: {result_b.get('completion_tokens', 'n/a')}"
        return "Both responses saved. Rate each model below or open History.", result_a["answer"], meta_a, result_b["answer"], meta_b, id_a, id_b
    except Exception as exc:
        return f"Model A succeeded; Model B failed: {exc}", result_a["answer"], meta_a, "", "", id_a, None


def save_rating(record_id: str | None, correctness: float, completeness: float, code_quality: float, notes: str) -> str:
    if not record_id:
        return "Run this model first, then save its rating."
    history = get_history()
    record = next((item for item in history if item.get("id") == record_id), None)
    if record is None:
        return "That evaluation is no longer in the local history. Run it again."
    record["ratings"] = {
        "correctness": int(correctness),
        "completeness": int(completeness),
        "code_quality": int(code_quality),
        "notes": (notes or "").strip(),
    }
    _write_json(HISTORY_PATH, history)
    return f"Rating saved for {record.get('model_id')} ({record.get('task_title')})."


def history_summary() -> list[dict[str, Any]]:
    rows = []
    for record in get_history()[:100]:
        ratings = record.get("ratings") or {}
        rows.append(
            {
                "date": record.get("created_at", ""),
                "task": record.get("task_title", ""),
                "category": record.get("category", ""),
                "model": record.get("model_id", ""),
                "seconds": record.get("seconds", ""),
                "correctness": ratings.get("correctness", ""),
                "completeness": ratings.get("completeness", ""),
                "code quality": ratings.get("code_quality", ""),
            }
        )
    return rows


def export_backup() -> str:
    backup = {"schema_version": 1, "tasks": get_tasks(), "history": get_history()}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="nova-codebench-backup-", delete=False, encoding="utf-8") as file:
        json.dump(backup, file, ensure_ascii=False, indent=2)
        return file.name


def import_backup(file_path: str | None) -> tuple[str, list[dict[str, Any]]]:
    if not file_path:
        return "Choose a JSON backup first.", history_summary()
    try:
        imported = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"Could not read that JSON file: {exc}", history_summary()

    # Accept the first app version's history-only backups as well as full backups.
    if isinstance(imported, list):
        imported_tasks: list[Any] = []
        imported_history = imported
    elif isinstance(imported, dict):
        imported_tasks = imported.get("tasks", [])
        imported_history = imported.get("history", [])
        if not isinstance(imported_tasks, list) or not isinstance(imported_history, list):
            return "The backup must contain task and history lists.", history_summary()
    else:
        return "The backup must be a JSON object created by Nova CodeBench.", history_summary()

    tasks = get_tasks()
    task_ids = {item.get("id") for item in tasks if isinstance(item, dict)}
    added_tasks = 0
    for item in imported_tasks:
        if isinstance(item, dict) and item.get("id") and item.get("title") and item.get("prompt") and item["id"] not in task_ids:
            tasks.append(item)
            task_ids.add(item["id"])
            added_tasks += 1
    _write_json(TASKS_PATH, tasks)

    current = get_history()
    known_ids = {item.get("id") for item in current}
    added = 0
    for item in imported_history:
        if isinstance(item, dict) and item.get("id") and item.get("response") is not None and item["id"] not in known_ids:
            current.append(item)
            known_ids.add(item["id"])
            added += 1
    current.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    _write_json(HISTORY_PATH, current)
    return f"Imported {added_tasks} task(s) and {added} evaluation record(s).", history_summary()


with gr.Blocks(title="Nova CodeBench", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Nova CodeBench\n"
        "Send coding tasks to a Hugging Face model, compare model versions, and keep a record of your ratings. "
        "This app calls your deployed model; it does not train the model."
    )

    latest_a = gr.State(None)
    latest_b = gr.State(None)

    with gr.Tab("Evaluate"):
        with gr.Row():
            with gr.Column(scale=1):
                task_title = gr.Textbox(label="Task title", placeholder="Example: Build a login API")
                category = gr.Dropdown(CATEGORIES, value="App creation", label="Task category")
                task_prompt = gr.Textbox(label="Coding task sent to the model", lines=8, placeholder="Describe the software or coding problem you want the model to solve.")
                rubric = gr.Textbox(label="What you will check (not sent to the model)", lines=3, placeholder="Expected features, tests, or acceptance criteria")
                system_prompt = gr.Textbox(
                    label="System instruction (sent to each model)",
                    value="You are a software engineering assistant. Produce correct, usable code. State assumptions briefly and include tests when they are helpful.",
                    lines=3,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Model A")
                mode_a = gr.Dropdown(
                    ["Hugging Face Inference Providers", "Hugging Face Inference Endpoint"],
                    value="Hugging Face Inference Endpoint",
                    label="Connection type",
                )
                model_id_a = gr.Textbox(label="Model ID / served model name", value=os.getenv("NOVA_MODEL_ID", ""), placeholder="your-hf-username/your-nova-model")
                endpoint_a = gr.Textbox(label="Endpoint URL (for Inference Endpoint mode)", value=os.getenv("NOVA_ENDPOINT_URL", ""), placeholder="https://your-endpoint.endpoints.huggingface.cloud")
                token_a = gr.Textbox(label="HF access token (optional if HF_TOKEN secret is set)", type="password", placeholder="hf_…")

                compare = gr.Checkbox(label="Compare with a second model", value=False)
                gr.Markdown("### Model B (optional comparison)")
                mode_b = gr.Dropdown(
                    ["Hugging Face Inference Providers", "Hugging Face Inference Endpoint"],
                    value="Hugging Face Inference Endpoint",
                    label="Connection type for Model B",
                )
                model_id_b = gr.Textbox(label="Model B ID / served model name", placeholder="another-model-or-version")
                endpoint_b = gr.Textbox(label="Model B endpoint URL (if using an endpoint)", placeholder="https://another-endpoint.endpoints.huggingface.cloud")
                token_b = gr.Textbox(label="Model B token (blank uses Model A token or HF_TOKEN)", type="password", placeholder="hf_…")
                temperature = gr.Slider(0, 1.5, value=0.2, step=0.1, label="Temperature")
                max_tokens = gr.Slider(128, 4096, value=1200, step=128, label="Maximum response tokens")

        run_button = gr.Button("Call model and evaluate", variant="primary")
        run_status = gr.Markdown()
        with gr.Row():
            with gr.Column():
                meta_a = gr.Markdown()
                response_a = gr.Textbox(label="Model A response", lines=18, interactive=False, show_copy_button=True)
            with gr.Column():
                meta_b = gr.Markdown()
                response_b = gr.Textbox(label="Model B response", lines=18, interactive=False, show_copy_button=True)

        gr.Markdown("### Rate the responses\nUse 1 for poor and 5 for strong. Ratings and responses are saved locally by the app.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Model A rating**")
                a_correct = gr.Slider(1, 5, value=3, step=1, label="Correctness")
                a_complete = gr.Slider(1, 5, value=3, step=1, label="Completeness")
                a_quality = gr.Slider(1, 5, value=3, step=1, label="Code quality")
                a_notes = gr.Textbox(label="Notes", lines=2)
                save_a = gr.Button("Save Model A rating")
                rating_status_a = gr.Markdown()
            with gr.Column():
                gr.Markdown("**Model B rating**")
                b_correct = gr.Slider(1, 5, value=3, step=1, label="Correctness")
                b_complete = gr.Slider(1, 5, value=3, step=1, label="Completeness")
                b_quality = gr.Slider(1, 5, value=3, step=1, label="Code quality")
                b_notes = gr.Textbox(label="Notes", lines=2)
                save_b = gr.Button("Save Model B rating")
                rating_status_b = gr.Markdown()

    with gr.Tab("Task library"):
        gr.Markdown("Save repeatable coding prompts and evaluation criteria. The rubric is kept out of the model prompt.")
        task_picker = gr.Dropdown(choices=_task_choices(), label="Saved task", value=None)
        load_button = gr.Button("Load selected task into Evaluate")
        library_status = gr.Markdown()
        with gr.Row():
            lib_title = gr.Textbox(label="Task title")
            lib_category = gr.Dropdown(CATEGORIES, value="App creation", label="Category")
        lib_prompt = gr.Textbox(label="Task prompt", lines=7)
        lib_rubric = gr.Textbox(label="Evaluation criteria (kept out of the model prompt)", lines=3)
        save_task_button = gr.Button("Save task")

    with gr.Tab("History and backup"):
        gr.Markdown("Evaluation results and saved tasks are stored in local JSON files. Download a backup regularly, especially when using a hosted Space.")
        refresh_history = gr.Button("Refresh history")
        history_table = gr.JSON(label="Recent evaluations", value=history_summary())
        download_history = gr.Button("Prepare full backup")
        history_file = gr.File(label="Download task and evaluation backup")
        import_file = gr.File(label="Restore or merge a Nova CodeBench JSON backup", file_types=[".json"], type="filepath")
        import_button = gr.Button("Import backup")
        history_status = gr.Markdown()

    run_button.click(
        run_evaluation,
        inputs=[task_title, category, task_prompt, rubric, system_prompt, mode_a, model_id_a, endpoint_a, token_a, compare, mode_b, model_id_b, endpoint_b, token_b, temperature, max_tokens],
        outputs=[run_status, response_a, meta_a, response_b, meta_b, latest_a, latest_b],
    )
    save_a.click(save_rating, inputs=[latest_a, a_correct, a_complete, a_quality, a_notes], outputs=rating_status_a)
    save_b.click(save_rating, inputs=[latest_b, b_correct, b_complete, b_quality, b_notes], outputs=rating_status_b)
    save_task_button.click(save_task, inputs=[lib_title, lib_category, lib_prompt, lib_rubric], outputs=[task_picker, library_status])
    load_button.click(load_task, inputs=task_picker, outputs=[task_title, category, task_prompt, rubric])
    refresh_history.click(history_summary, outputs=history_table)
    download_history.click(export_backup, outputs=history_file)
    import_button.click(import_backup, inputs=import_file, outputs=[history_status, history_table])


if __name__ == "__main__":
    demo.queue().launch()
