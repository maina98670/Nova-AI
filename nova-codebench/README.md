---
title: Nova CodeBench
emoji: 🧪
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Nova CodeBench

Nova CodeBench is a small evaluation dashboard for your coding model. It sends coding tasks to a Hugging Face model, shows responses side by side, lets you rate them, and saves the evaluation history.

The app calls a model that is already available through Hugging Face inference. It does not train or host the model weights itself.

## What it does

- Calls a model through Hugging Face Inference Providers or an OpenAI-compatible Hugging Face Inference Endpoint.
- Compares two model IDs or endpoint deployments on the same task.
- Saves repeatable coding tasks with evaluation criteria that are not sent to the model.
- Records correctness, completeness, code-quality ratings, latency, and notes.
- Exports and imports JSON backups of the local task library and evaluation history.

## Hugging Face setup

### Inference Providers

Choose **Hugging Face Inference Providers** in the app, enter a model ID available through a compatible provider, and provide a Hugging Face access token with Inference Providers permission. The app sends chat-completion requests to Hugging Face's router API.

### Your own deployed model

For a fine-tuned Nova model, upload the model artifact to the Hugging Face Hub and deploy it as an Inference Endpoint. Choose an engine that exposes an OpenAI-compatible chat-completion route, then select **Hugging Face Inference Endpoint** and enter the endpoint URL and the model name expected by that endpoint. If the endpoint URL already ends in `/v1` or `/v1/chat/completions`, the app uses that path; otherwise it appends `/v1/chat/completions`.

## Run on your computer

Install Python 3.10 or newer, then run:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio address printed in the terminal. You can enter a token in the password field, or set `HF_TOKEN` in the environment before starting the app. The token is not written to the app's JSON files.

## Host as a Hugging Face Space

Create a **private Gradio Space**, upload `app.py`, `requirements.txt`, and this `README.md`, then add `HF_TOKEN` under the Space's **Settings → Secrets**. In the app, enter the model ID. The app reads the secret on the server when the token field is blank. Keep the Space private when it uses your token, because anyone who can use a public app could otherwise send requests through your account.

## Data and privacy

Prompts, model responses, and ratings are saved to `data/tasks.json` and `data/history.json` in the app's local directory. API tokens are not saved. Prompts are sent to Hugging Face when you run an evaluation. Download a history backup before rebuilding or moving a hosted Space; local Space files may not be a durable backup.

This first version displays model responses for you to inspect and rate. It does not execute generated code. Run code from model responses only in a sandbox or a disposable development environment.
