# Nova AI

This repository contains Nova AI development projects.

## Nova CodeBench

The `nova-codebench/` directory contains a Gradio evaluation app that calls coding models through Hugging Face Inference Providers or an OpenAI-compatible Hugging Face Inference Endpoint. It records tasks, model outputs, and human ratings.

See [Nova CodeBench setup and deployment instructions](nova-codebench/README.md). Do not commit Hugging Face access tokens; configure them as private secrets in the deployment environment.
