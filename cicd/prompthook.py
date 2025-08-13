import base64
import json
import uuid
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Configuration ---
class AppConfig(BaseSettings):
    """
    Application configuration model.
    Loads settings from environment variables.
    """
    GITHUB_TOKEN: str
    GITHUB_REPO_OWNER: str
    GITHUB_REPO_NAME: str
    GITHUB_FILE_PATH: str = "cicd/prompt_manifest.json"
    GITHUB_BRANCH: str = "main"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = AppConfig()

# --- Pydantic Models ---
class WebhookPayload(BaseModel):
    """
    Defines the expected structure of the incoming webhook payload.
    """
    prompt_id: uuid.UUID = Field(
        ...,
        description="The unique identifier for the prompt."
    )
    prompt_name: str = Field(
        ...,
        description="The name/title of the prompt."
    )
    commit_hash: str = Field(
        ...,
        description="An identifier for the commit event that triggered the webhook."
    )
    created_at: str = Field(
        ...,
        description="Timestamp indicating when the event was created (ISO format preferred)."
    )
    created_by: str = Field(
        ...,
        description="The name of the user who created the event."
    )
    manifest: Dict[str, Any] = Field(
        ...,
        description="The main content or configuration data to be committed to GitHub."
    )

# --- GitHub Helper Function ---
async def commit_manifest_to_github(payload: WebhookPayload) -> Dict[str, Any]:
    """
    Helper function to commit the manifest directly to the configured branch.
    """
    github_api_base_url = "https://api.github.com"
    repo_file_url = (
        f"{github_api_base_url}/repos/{settings.GITHUB_REPO_OWNER}/"
        f"{settings.GITHUB_REPO_NAME}/contents/{settings.GITHUB_FILE_PATH}"
    )

    headers = {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    manifest_json_string = json.dumps(payload.manifest, indent=2)
    content_base64 = base64.b64encode(manifest_json_string.encode('utf-8')).decode('utf-8')
    commit_message = f"feat: Update {settings.GITHUB_FILE_PATH} via webhook - commit {payload.commit_hash}"

    data_to_commit = {
        "message": commit_message,
        "content": content_base64,
        "branch": settings.GITHUB_BRANCH,
    }

    async with httpx.AsyncClient() as client:
        current_file_sha = None
        try:
            params_get = {"ref": settings.GITHUB_BRANCH}
            response_get = await client.get(repo_file_url, headers=headers, params=params_get)
            if response_get.status_code == 200:
                current_file_sha = response_get.json().get("sha")
            elif response_get.status_code != 404: # If not 404 (not found), it's an unexpected error
                response_get.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_detail = f"GitHub API error (GET file SHA): {e.response.status_code} - {e.response.text}"
            print(f"[ERROR] {error_detail}")
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except httpx.RequestError as e:
            error_detail = f"Network error connecting to GitHub (GET file SHA): {str(e)}"
            print(f"[ERROR] {error_detail}")
            raise HTTPException(status_code=503, detail=error_detail)

        if current_file_sha:
            data_to_commit["sha"] = current_file_sha

        try:
            print(f"PUT URL: {repo_file_url}")
            print(f"File path: {settings.GITHUB_FILE_PATH}")
            print(f"Branch: {settings.GITHUB_BRANCH}")
            print(f"Repo owner: {settings.GITHUB_REPO_OWNER}")
            print(f"Repo name: {settings.GITHUB_REPO_NAME}")
            response_put = await client.put(repo_file_url, headers=headers, json=data_to_commit)
            response_put.raise_for_status()
            return response_put.json()
        except httpx.HTTPStatusError as e:
            error_detail = f"GitHub API error (PUT content): {e.response.status_code} - {e.response.text}"
            if e.response.status_code == 409: # Conflict
                error_detail = (
                    f"GitHub API conflict (PUT content): {e.response.text}. "
                    "This might be due to an outdated SHA or branch protection rules."
                )
            elif e.response.status_code == 422: # Unprocessable Entity
                 error_detail = (
                    f"GitHub API Unprocessable Entity (PUT content): {e.response.text}. "
                    f"Ensure the branch '{settings.GITHUB_BRANCH}' exists and the payload is correctly formatted."
                )
            print(f"[ERROR] {error_detail}")
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except httpx.RequestError as e:
            error_detail = f"Network error connecting to GitHub (PUT content): {str(e)}"
            print(f"[ERROR] {error_detail}")
            raise HTTPException(status_code=503, detail=error_detail)

# --- FastAPI Application ---
app = FastAPI(
    title="Minimal Webhook to GitHub Commit Service",
    description="Receives a webhook and commits its 'manifest' part directly to a GitHub repository.",
    version="0.1.0",
)

@app.post("/webhook/commit", status_code=201, tags=["GitHub Webhooks"])
async def handle_webhook_direct_commit(payload: WebhookPayload = Body(...)):
    """
    Webhook endpoint to receive events and commit DIRECTLY to the configured branch.
    """
    try:
        github_response = await commit_manifest_to_github(payload)
        return {
            "message": "Webhook received and manifest committed directly to GitHub successfully.",
            "github_commit_details": github_response.get("commit", {}),
            "github_content_details": github_response.get("content", {})
        }
    except HTTPException:
        raise # Re-raise if it's an HTTPException from the helper
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(f"[ERROR] {error_message}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/health", status_code=200, tags=["Health"])
async def health_check():
    """
    A simple health check endpoint.
    """
    return {"status": "ok", "message": "Service is running."}

# To run this server:
# 1. Ensure your .env file contains your GitHub token and repo details.
# 2. Run with Uvicorn: uvicorn cicd.prompthook:app --reload
# 3. Deploy to a public platform like Render.com.