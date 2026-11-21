import asyncio
import base64
import html as html_lib
import io
import json
import os
import re
import secrets
import smtplib
import threading
import time
import uuid
import wave
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Literal, Optional

import edge_tts
import httpx
import jwt
import pymupdf
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pwdlib import PasswordHash
from pydantic import BaseModel, EmailStr, field_validator
from supabase import Client, create_client

load_dotenv()

# ==========================================================
# CONFIGURATION (.env)
# ==========================================================
SUPABASE_URL = os.environ["SUPABASE_URL"].strip().rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip()
ADMIN_USERNAME = os.environ["ADMIN_USERNAME"].strip()
ADMIN_PASSWORD_HASH = os.environ["ADMIN_PASSWORD_HASH"].strip()
JWT_SECRET = os.environ["JWT_SECRET"].strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b").strip()
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()

SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY", "").strip()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
GMAIL_USER = os.getenv("GMAIL_USER", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "").strip()
MOVIES_SITE_URL = os.getenv("MOVIES_SITE_URL", "").strip()
MOVIES_SITE_PASSWORD = os.getenv("MOVIES_SITE_PASSWORD", "").strip()

MAX_HISTORY_MESSAGES = 15  # Rolling memory window for every AI tool

TEXT_FALLBACK_MODELS = list(
    dict.fromkeys(
        [
            GROQ_TEXT_MODEL,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
    )
)

FAST_TEXT_MODELS = list(
    dict.fromkeys(
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
        ]
    )
)

VISION_FALLBACK_MODELS = list(
    dict.fromkeys(
        [
            GROQ_VISION_MODEL,
            "qwen/qwen3.8-27b",
        ]
    )
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY, timeout=45.0, max_retries=1) if GROQ_API_KEY else None
password_hash = PasswordHash.recommended()
security = HTTPBearer()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
NOTES_DB_FILE = BASE_DIR / "notes_db.json"
YT_CACHE_FILE = BASE_DIR / "yt_cache.json"
MEMORY_DB_FILE = BASE_DIR / "ai_memory.json"
TG_USERS_FILE = BASE_DIR / "tg_users.json"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)


# ==========================================================
# LIFESPAN (AUTO-STARTS TELEGRAM BOT AUTOMATION WORKER)
# ==========================================================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    bot_task = asyncio.create_task(telegram_bot_polling_loop())
    try:
        yield
    finally:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="StudyDesk Unified Backend (Groq + Telegram Automation)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# ==========================================================
# 15-MESSAGE CONVERSATION MEMORY MANAGER (ALL AI TOOLS)
# ==========================================================
memory_lock = threading.Lock()


def _load_memory_store() -> dict[str, list[dict]]:
    if not MEMORY_DB_FILE.exists():
        return {}
    try:
        data = json.loads(MEMORY_DB_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


ai_memory_store: dict[str, list[dict]] = _load_memory_store()


def _persist_memory_store():
    try:
        MEMORY_DB_FILE.write_text(json.dumps(ai_memory_store, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[Memory] Save warning: {e}", flush=True)


def get_session_id(request: Request, explicit_session: str = "") -> str:
    raw = (explicit_session or request.headers.get("x-session-id") or "").strip()
    if not raw:
        raw = request.client.host if request.client else "default"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)[:64]
    return cleaned or "default"


def _mem_key(session_id: str, tool_key: str) -> str:
    safe_tool = re.sub(r"[^a-zA-Z0-9._:-]", "_", tool_key.lower().strip())[:64]
    return f"{session_id}::{safe_tool}"


def get_tool_history(session_id: str, tool_key: str) -> list[dict]:
    key = _mem_key(session_id, tool_key)
    with memory_lock:
        return list(ai_memory_store.get(key, [])[-MAX_HISTORY_MESSAGES:])


def append_tool_history(session_id: str, tool_key: str, user_text: str, assistant_text: str):
    key = _mem_key(session_id, tool_key)
    now_str = datetime.now().strftime("%I:%M %p")
    u_clean = (user_text or "").strip()[:2200]
    a_clean = (assistant_text or "").strip()[:2600]
    with memory_lock:
        hist = ai_memory_store.get(key, [])
        if u_clean:
            hist.append({"role": "user", "content": u_clean, "time": now_str})
        if a_clean:
            hist.append({"role": "assistant", "content": a_clean, "time": now_str})
        ai_memory_store[key] = hist[-MAX_HISTORY_MESSAGES:]
        _persist_memory_store()


def clear_tool_history(session_id: str, tool_key: str = "all") -> int:
    removed = 0
    with memory_lock:
        if tool_key.lower().strip() == "all":
            prefix = f"{session_id}::"
            keys = [k for k in list(ai_memory_store.keys()) if k.startswith(prefix)]
            for k in keys:
                del ai_memory_store[k]
                removed += 1
        else:
            key = _mem_key(session_id, tool_key)
            if key in ai_memory_store:
                del ai_memory_store[key]
                removed = 1
            sub_prefix = key + ":"
            for k in [x for x in list(ai_memory_store.keys()) if x.startswith(sub_prefix)]:
                del ai_memory_store[k]
                removed += 1
        _persist_memory_store()
    return removed


# ==========================================================
# CORE HELPERS (RATE LIMIT, TELEGRAM + GMAIL/BREVO, GROQ AI)
# ==========================================================
rate_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(key: str, max_attempts: int = 30, window_seconds: int = 300):
    now = time.time()
    recent = [t for t in rate_store[key] if now - t < window_seconds]
    if len(recent) >= max_attempts:
        raise HTTPException(status_code=429, detail="Too many tries. Please wait 2 minutes.")
    recent.append(now)
    rate_store[key] = recent


def load_tg_users() -> dict:
    if not TG_USERS_FILE.exists():
        return {}
    try:
        data = json.loads(TG_USERS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_tg_user(identifier: str, chat_id: str):
    key = identifier.strip().lower()
    if not key:
        return
    data = load_tg_users()
    data[key] = str(chat_id)
    TG_USERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def send_telegram(message: str, chat_id: Optional[str] = None, reply_markup: Optional[dict] = None) -> bool:
    target = str(chat_id or TELEGRAM_CHAT_ID).strip()
    if not TELEGRAM_BOT_TOKEN or not target:
        return False
    payload: dict = {"chat_id": target, "text": message, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=10.0,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[Telegram] Error: {e}", flush=True)
        return False


def send_student_telegram(email: str, phone: str, message: str) -> bool:
    """Sends OTP or payment status directly to the student on Telegram if they linked the bot once."""
    users = load_tg_users()
    digits = re.sub(r"\D", "", phone or "")[-10:]
    chat_id = users.get((email or "").lower().strip()) or (users.get(digits) if digits else None)
    if not chat_id:
        return False
    return send_telegram(message, chat_id=chat_id)


def send_email(to_email: str, subject: str, body: str) -> bool:
    # 1. Brevo HTTP API (free 300 emails/day, works on Render free tier & locally)
    if BREVO_API_KEY and GMAIL_USER:
        try:
            r = httpx.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
                json={
                    "sender": {"name": "StudyDesk", "email": GMAIL_USER},
                    "to": [{"email": to_email}],
                    "subject": subject,
                    "textContent": body,
                },
                timeout=12.0,
            )
            if r.status_code in (200, 201, 202):
                return True
            print(f"[Brevo Email] HTTP {r.status_code}: {r.text[:120]}", flush=True)
        except Exception as e:
            print(f"[Brevo Email] Error: {e}", flush=True)

    # 2. Gmail SMTP fallback (works locally on Windows laptop)
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=12) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"[Gmail SMTP] Error: {e}", flush=True)
        return False


def require_ai():
    if not groq_client:
        raise HTTPException(status_code=500, detail="Add GROQ_API_KEY in backend/.env and restart uvicorn.")


def strip_thinking_tags(text: str) -> str:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE)
    return cleaned.strip()


def clean_readable_math(text: str) -> str:
    """Removes Markdown & LaTeX symbols so answers read cleanly like a notebook."""
    s = strip_thinking_tags(text)

    for _ in range(3):
        s = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", s)
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", s)
        s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathit|operatorname)\{([^{}]+)\}", r"\1", s)

    replacements = [
        (r"\\times", "×"),
        (r"\\cdot", "×"),
        (r"\\div", "÷"),
        (r"\\pm", "±"),
        (r"\\mp", "∓"),
        (r"\\leq|\\le", "<="),
        (r"\\geq|\\ge", ">="),
        (r"\\neq|\\ne", "!="),
        (r"\\approx", "≈"),
        (r"\\infty", "∞"),
        (r"\\pi", "π"),
        (r"\\theta", "θ"),
        (r"\\alpha", "α"),
        (r"\\beta", "β"),
        (r"\\gamma", "γ"),
        (r"\\Delta|\\delta", "Δ"),
        (r"\\lambda", "λ"),
        (r"\\mu", "μ"),
        (r"\\omega", "ω"),
        (r"\\Omega", "Ω"),
        (r"\\degree|\^\\circ", "°"),
        (r"\\Rightarrow|\\implies", "=>"),
        (r"\\rightarrow|\\to", "->"),
        (r"\\left|\\right|\\quad|\\qquad|\\,|\\;|\\!", ""),
        (r"\^2\b|\^\{2\}", "²"),
        (r"\^3\b|\^\{3\}", "³"),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)

    s = re.sub(r"\$\$|\$|\\\[|\\\]|\\\(|\\\)", "", s)
    s = re.sub(r"```[a-zA-Z]*", "", s)
    s = s.replace("```", "").replace("`", "")
    s = re.sub(r"\*\*([^\n*]+)\*\*", r"\1", s)
    s = re.sub(r"__([^\n_]+)__", r"\1", s)
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", s)
    s = re.sub(r"(?m)^\s*[-_=]{3,}\s*$", "", s)
    s = s.replace("**", "").replace("*", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"\{([^{}]{1,40})\}", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_speech_text(text: str) -> str:
    s = clean_readable_math(text)
    s = re.sub(r"[•#_~>\[\]|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:4500]


def prepare_image_data_url(img_bytes: bytes, mime: str = "image/jpeg") -> str:
    safe_mime = mime if mime in ("image/jpeg", "image/png", "image/webp") else "image/jpeg"
    payload = img_bytes

    if len(img_bytes) > 2_000_000:
        try:
            ext = safe_mime.split("/")[-1]
            doc = pymupdf.open(stream=img_bytes, filetype=ext)
            page = doc[0]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(0.65, 0.65))
            payload = pix.tobytes("jpeg", jpg_quality=80)
            safe_mime = "image/jpeg"
            doc.close()
        except Exception as e:
            print(f"[Image Compress] Using original bytes: {e}", flush=True)

    b64 = base64.b64encode(payload).decode("utf-8")
    return f"data:{safe_mime};base64,{b64}"


def _build_messages(
    current_content,
    system_prompt: str = "",
    history: Optional[list[dict]] = None,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        for item in history[-MAX_HISTORY_MESSAGES:]:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content[:2500]})
    messages.append({"role": "user", "content": current_content})
    return messages


def _run_groq_sync(
    messages: list,
    models: list[str],
    json_mode: bool = False,
    max_completion_tokens: int = 4096,
    reasoning_effort: str = "low",
) -> str:
    require_ai()
    last_err = None

    for model_id in models:
        kwargs = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.4 if json_mode else 0.65,
            "max_completion_tokens": max_completion_tokens,
        }

        if model_id.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["include_reasoning"] = False
        elif model_id.startswith("qwen/qwen3"):
            kwargs["reasoning_effort"] = "none" if json_mode else "low"
            kwargs["reasoning_format"] = "hidden"

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(2):
            try:
                resp = groq_client.chat.completions.create(**kwargs)
                content = strip_thinking_tags(resp.choices[0].message.content or "")
                if content:
                    return content
                raise ValueError("Empty content returned by model.")
            except Exception as e:
                err_str = str(e)
                last_err = e
                print(f"[Groq] {model_id} attempt {attempt + 1} error: {err_str[:140]}", flush=True)

                if "401" in err_str or "invalid_api_key" in err_str.lower():
                    raise HTTPException(status_code=401, detail="Invalid GROQ_API_KEY in backend/.env.")

                if (
                    "400" in err_str
                    or "response_format" in err_str
                    or "reasoning" in err_str
                    or "json_validate_failed" in err_str
                ):
                    kwargs.pop("response_format", None)
                    kwargs.pop("reasoning_format", None)
                    kwargs.pop("include_reasoning", None)
                    kwargs.pop("reasoning_effort", None)
                    continue

                if "404" in err_str or "model_not_found" in err_str.lower() or "decommissioned" in err_str.lower():
                    break
                time.sleep(0.25)

    raise HTTPException(
        status_code=503,
        detail=f"Groq AI request failed. Check GROQ_API_KEY or retry. ({str(last_err)[:130]})",
    )


async def ask_groq(
    prompt: str,
    system_prompt: str = "",
    history: Optional[list[dict]] = None,
    json_mode: bool = False,
    fast: bool = False,
    max_completion_tokens: int = 4096,
) -> str:
    models = FAST_TEXT_MODELS if fast else TEXT_FALLBACK_MODELS
    messages = _build_messages(prompt, system_prompt=system_prompt, history=history)
    return await asyncio.to_thread(_run_groq_sync, messages, models, json_mode, max_completion_tokens, "low")


async def ask_groq_vision(
    prompt: str,
    images: list[tuple[bytes, str]],
    system_prompt: str = "",
    history: Optional[list[dict]] = None,
    json_mode: bool = False,
    max_completion_tokens: int = 4096,
) -> str:
    content_blocks = [{"type": "text", "text": prompt}]
    for img_bytes, mime in images[:3]:
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": prepare_image_data_url(img_bytes, mime)},
            }
        )
    messages = _build_messages(content_blocks, system_prompt=system_prompt, history=history)
    return await asyncio.to_thread(_run_groq_sync, messages, VISION_FALLBACK_MODELS, json_mode, max_completion_tokens, "low")


def extract_json(raw: str, fallback: Optional[dict] = None) -> dict:
    cleaned = strip_thinking_tags(raw)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    if fallback is not None:
        return fallback
    raise HTTPException(status_code=500, detail="AI returned an unexpected format. Please try again.")


# ---------------- AUDIO / TTS HELPERS ----------------
def _groq_orpheus_line_sync(text: str, voice: str = "diana") -> Optional[bytes]:
    if not groq_client:
        return None
    try:
        resp = groq_client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice,
            input=text[:190],
            response_format="wav",
        )
        return resp.read()
    except Exception as e:
        print(f"[Groq Orpheus] Unavailable/Skipped: {str(e)[:100]}", flush=True)
        return None


def _gtts_save_sync(text: str, out_path: Path, lang: str = "en") -> bool:
    try:
        from gtts import gTTS

        tts = gTTS(text=text[:3500], lang=lang, tld="co.in" if lang == "en" else "com")
        tts.save(str(out_path))
        return out_path.exists() and out_path.stat().st_size > 200
    except Exception as e:
        print(f"[gTTS] Fallback error: {e}", flush=True)
        return False


async def generate_short_tts(text: str, prefix: str = "eng", lang: str = "en") -> str:
    speech = clean_speech_text(text)
    if not speech:
        return ""

    if lang == "en":
        wav_bytes = await asyncio.to_thread(_groq_orpheus_line_sync, speech, "diana")
        if wav_bytes:
            wav_name = f"{prefix}_{uuid.uuid4().hex[:10]}.wav"
            (AUDIO_DIR / wav_name).write_bytes(wav_bytes)
            return f"/audio/{wav_name}"

    mp3_name = f"{prefix}_{uuid.uuid4().hex[:10]}.mp3"
    mp3_path = AUDIO_DIR / mp3_name
    voice = "en-IN-NeerjaNeural" if lang == "en" else "hi-IN-SwaraNeural"
    try:
        communicate = edge_tts.Communicate(speech[:1000], voice)
        await asyncio.wait_for(communicate.save(str(mp3_path)), timeout=6.0)
        if mp3_path.exists() and mp3_path.stat().st_size > 200:
            return f"/audio/{mp3_name}"
    except Exception as e:
        print(f"[Edge-TTS] Timed out/failed ({e}), switching to gTTS...", flush=True)

    if await asyncio.to_thread(_gtts_save_sync, speech[:1200], mp3_path, lang):
        return f"/audio/{mp3_name}"
    return ""


# ---------------- YOUTUBE MULTI-LAYER EXTRACTOR ----------------
def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Unauthorized")
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid login token.")


def load_notes_db() -> list:
    if not NOTES_DB_FILE.exists():
        return []
    try:
        return json.loads(NOTES_DB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_notes_db(items: list):
    NOTES_DB_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def load_yt_cache() -> dict:
    if not YT_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(YT_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_yt_cache(video_id: str, text: str):
    cache = load_yt_cache()
    cache[video_id] = text[:20000]
    YT_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def try_extract_youtube_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pat in patterns:
        match = re.search(pat, url.strip())
        if match:
            return match.group(1)
    return None


def fetch_yt_transcript_lib_sync(video_id: str) -> Optional[str]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        if hasattr(api, "list"):
            tlist = api.list(video_id)
            chosen = None
            try:
                chosen = tlist.find_transcript(["en", "en-IN", "hi", "en-US", "en-GB"])
            except Exception:
                for t in tlist:
                    chosen = t
                    break
            if chosen:
                fetched = chosen.fetch()
                return " ".join(getattr(s, "text", str(s)) for s in fetched).strip()

        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                chosen = tlist.find_transcript(["en", "en-IN", "hi", "en-US", "en-GB"])
            except Exception:
                chosen = next(iter(tlist))
            return " ".join(row["text"] for row in chosen.fetch()).strip()
    except Exception as e:
        print(f"[YT Layer 1 - Lib] {e}", flush=True)
    return None


def fetch_yt_webpage_data_sync(video_id: str) -> dict:
    result = {"transcript": None, "title": "", "author": "", "description": ""}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    }
    try:
        r = httpx.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers, timeout=12.0, follow_redirects=True)
        if r.status_code != 200:
            return result
        page = r.text

        title_m = re.search(r'"title":"([^"]{3,220})"', page)
        if title_m:
            result["title"] = html_lib.unescape(title_m.group(1).encode("utf-8").decode("unicode_escape", errors="ignore"))
        desc_m = re.search(r'"shortDescription":"((?:\\.|[^"\\]){10,2500})"', page)
        if desc_m:
            result["description"] = desc_m.group(1).replace("\\n", "\n").replace('\\"', '"')

        tracks_m = re.search(r'"captionTracks":(\[.*?\])', page)
        if tracks_m:
            tracks = json.loads(tracks_m.group(1))
            chosen_url = None
            for pref in ("en", "en-IN", "hi"):
                for tr in tracks:
                    if str(tr.get("languageCode", "")).startswith(pref) and tr.get("baseUrl"):
                        chosen_url = tr["baseUrl"]
                        break
                if chosen_url:
                    break
            if not chosen_url and tracks:
                chosen_url = tracks[0].get("baseUrl")

            if chosen_url:
                cap_res = httpx.get(chosen_url, headers=headers, timeout=10.0)
                if cap_res.status_code == 200 and cap_res.text.strip():
                    root = ET.fromstring(cap_res.text)
                    lines = [html_lib.unescape("".join(el.itertext())).strip() for el in root.findall(".//text")]
                    joined = " ".join(x for x in lines if x)
                    if len(joined) > 40:
                        result["transcript"] = joined
    except Exception as e:
        print(f"[YT Layer 2 - Page Scrape] {e}", flush=True)
    return result


def fetch_supadata_transcript_sync(video_id: str) -> Optional[str]:
    if not SUPADATA_API_KEY:
        return None
    try:
        r = httpx.get(
            "https://api.supadata.ai/v1/transcript",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "text": "true"},
            headers={"x-api-key": SUPADATA_API_KEY},
            timeout=20.0,
        )
        if r.status_code == 200:
            content = r.json().get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                return " ".join(str(seg.get("text", "")) for seg in content).strip()
    except Exception as e:
        print(f"[YT Layer 3 - Supadata] {e}", flush=True)
    return None


def transcribe_yt_via_groq_whisper_sync(video_id: str) -> Optional[str]:
    if not groq_client:
        return None
    try:
        import yt_dlp
    except ImportError:
        return None

    temp_stem = AUDIO_DIR / f"yt_{uuid.uuid4().hex[:8]}"
    downloaded_file = None
    try:
        ydl_opts = {
            "format": "worstaudio[filesize<24M]/worstaudio[ext=m4a]/worstaudio",
            "outtmpl": f"{temp_stem}.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "max_filesize": 24 * 1024 * 1024,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            downloaded_file = Path(ydl.prepare_filename(info))

        if not downloaded_file.exists() or downloaded_file.stat().st_size > 25 * 1024 * 1024:
            return None

        with open(downloaded_file, "rb") as audio_fp:
            result = groq_client.audio.transcriptions.create(
                file=(downloaded_file.name, audio_fp.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        return str(result).strip()
    except Exception as e:
        print(f"[YT Layer 4 - Groq Whisper] {e}", flush=True)
        return None
    finally:
        if downloaded_file and downloaded_file.exists():
            try:
                downloaded_file.unlink()
            except Exception:
                pass


def fetch_yt_oembed_sync(video_id: str) -> dict:
    try:
        r = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=8.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@app.get("/")
@app.get("/api/health")
def health():
    return {
        "status": "StudyDesk Backend (Groq + Telegram Automation) is running!",
        "groq_key_loaded": bool(GROQ_API_KEY),
        "telegram_bot_loaded": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "email_ready": bool((BREVO_API_KEY and GMAIL_USER) or (GMAIL_USER and GMAIL_APP_PASSWORD)),
        "memory_window": MAX_HISTORY_MESSAGES,
        "text_models": TEXT_FALLBACK_MODELS,
        "vision_models": VISION_FALLBACK_MODELS,
    }


# ---------------- MEMORY VIEW / RESET ENDPOINTS ----------------
class ClearMemoryReq(BaseModel):
    tool: str = "all"
    session_id: str = ""


@app.get("/api/tools/history/{tool_key}")
def api_get_tool_history(tool_key: str, request: Request, scenario: str = ""):
    sid = get_session_id(request)
    full_key = f"{tool_key}:{scenario}" if scenario else tool_key
    hist = get_tool_history(sid, full_key)
    return {"tool": full_key, "count": len(hist), "history": hist}


@app.post("/api/tools/clear-memory")
def api_clear_tool_memory(req: ClearMemoryReq, request: Request):
    sid = get_session_id(request, req.session_id)
    cleared = clear_tool_history(sid, req.tool)
    return {"message": f"Cleared memory ({cleared} channel(s)).", "cleared": cleared}


# ==========================================================
# 1. STUDENT SIGN-UP / LOGIN OTP (AUTOMATED GMAIL + TELEGRAM)
# ==========================================================
otp_store: dict[str, dict] = {}


class AuthOTPReq(BaseModel):
    name: str
    phone: str
    email: EmailStr

    @field_validator("phone")
    @classmethod
    def check_phone(cls, v: str) -> str:
        digits = re.sub(r"\D", "", v)
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) != 10:
            raise ValueError("Enter a valid 10-digit mobile number.")
        return digits


class AuthOTPCheck(BaseModel):
    email: EmailStr
    otp: str


@app.post("/api/auth/send-otp")
def auth_send_otp(req: AuthOTPReq, request: Request):
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"auth_otp:{ip}", max_attempts=20, window_seconds=300)

    email = req.email.lower().strip()
    name = req.name.strip() or "Student"
    otp = str(secrets.randbelow(900000) + 100000)

    otp_store[email] = {
        "otp": otp,
        "name": name,
        "phone": req.phone,
        "expires": datetime.now(timezone.utc) + timedelta(minutes=10),
    }

    email_ok = send_email(
        email,
        f"Your StudyDesk Login OTP: {otp}",
        f"Hi {name},\n\nYour 6-digit StudyDesk verification OTP is: {otp}\nMobile: {req.phone}\nValid for 10 minutes.\n\n— StudyDesk",
    )
    student_tg_ok = send_student_telegram(
        email,
        req.phone,
        f"<b>StudyDesk Login OTP</b>\nHi {name}, your 6-digit OTP is: <code>{otp}</code>\nValid for 10 minutes.",
    )
    admin_tg_ok = send_telegram(
        f"<b>Student Login OTP Issued</b>\nName: {name}\nMobile: <code>{req.phone}</code>\n"
        f"Gmail: <code>{email}</code>\n6-Digit OTP: <code>{otp}</code>"
    )
    print(f"\n[STUDENT OTP] Name={name} | Mobile={req.phone} | Gmail={email} | OTP={otp}\n", flush=True)

    if email_ok and student_tg_ok:
        return {"message": f"6-digit OTP sent automatically to {email} and Telegram!"}
    if email_ok:
        return {"message": f"6-digit OTP sent automatically to {email}!"}
    if student_tg_ok:
        return {"message": "6-digit OTP sent automatically to your Telegram!"}
    if admin_tg_ok:
        return {"message": "OTP generated and logged to Telegram Bot!"}
    return {"message": "OTP generated! Check your VS Code terminal."}


@app.post("/api/auth/verify-otp")
def auth_verify_otp(req: AuthOTPCheck):
    email = req.email.lower().strip()
    rec = otp_store.get(email)
    if not rec or datetime.now(timezone.utc) > rec["expires"]:
        raise HTTPException(status_code=400, detail="OTP expired or not found. Click Send OTP again.")
    if not secrets.compare_digest(rec["otp"], req.otp.strip()):
        raise HTTPException(status_code=400, detail="Incorrect 6-digit OTP. Please try again.")
    user = {"name": rec["name"], "phone": rec["phone"], "email": email}
    del otp_store[email]
    return {"message": "OTP verified! Welcome to StudyDesk.", "user": user}


# ==========================================================
# 1B. PROTECTED PRO BONUS VAULT (FREE MOVIES WEBSITE)
# ==========================================================
@app.get("/api/pro/movies-pass")
def get_movies_pass(email: str = "", xp_pro: bool = False):
    allowed = xp_pro
    clean_email = email.lower().strip()

    if clean_email and not allowed:
        try:
            res = supabase.table("subscriptions").select("plan,active_until").eq("email", clean_email).limit(1).execute()
            if res.data:
                exp = datetime.fromisoformat(res.data[0]["active_until"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) <= exp:
                    allowed = True
        except Exception as e:
            print(f"[Pro Perks] Supabase check error: {e}", flush=True)

    if not allowed:
        raise HTTPException(status_code=403, detail="Active Pro Plan required to view the Free Movies Website password.")

    return {
        "site_url": MOVIES_SITE_URL or "Link coming soon",
        "password": MOVIES_SITE_PASSWORD or "Password not set in .env yet",
    }


# ==========================================================
# 2. QR PAYMENT, 1-TAP TELEGRAM APPROVAL & PRO ACTIVATION OTP
# ==========================================================
class PaymentSubmit(BaseModel):
    name: str = "Student"
    email: EmailStr
    phone: str
    utr: str
    amount: int = 99

    @field_validator("phone")
    @classmethod
    def check_phone(cls, v: str) -> str:
        digits = re.sub(r"\D", "", v)
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) != 10:
            raise ValueError("Enter a valid 10-digit mobile number.")
        return digits

    @field_validator("utr")
    @classmethod
    def check_utr(cls, v: str) -> str:
        cleaned = re.sub(r"\D", "", v)
        if len(cleaned) != 12:
            raise ValueError("UPI UTR must be exactly 12 digits.")
        return cleaned


class PaymentOTPVerify(BaseModel):
    email: EmailStr
    otp_code: str

    @field_validator("otp_code")
    @classmethod
    def check_otp(cls, v: str) -> str:
        cleaned = v.strip()
        if not re.fullmatch(r"\d{6}", cleaned):
            raise ValueError("OTP must be exactly 6 digits.")
        return cleaned


class AdminLogin(BaseModel):
    username: str
    password: str


class AdminDecision(BaseModel):
    request_id: str
    action: Literal["approve", "reject"]


def process_payment_decision(request_id: str, action: str) -> dict:
    """Shared automated decision handler for both Telegram 1-tap buttons and Admin Panel."""
    res = supabase.table("payment_requests").select("*").eq("id", request_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Payment request not found.")
    row = res.data[0]
    if row["status"] != "PENDING":
        return {"already_done": True, "status": row["status"], "message": f"Already {row['status']}."}

    student_email = row["email"]
    student_phone = row.get("phone", "")
    amount = row.get("amount", 99)

    if action == "reject":
        supabase.table("payment_requests").update({"status": "REJECTED"}).eq("id", request_id).execute()
        reject_msg = (
            "You haven't done the payment yet.\n\n"
            f"Your payment claim (UTR: {row.get('utr')}) could not be verified. "
            "Please do the UPI payment first and submit your valid 12-digit UTR on StudyDesk — "
            "then your 6-digit Pro activation OTP will be sent to you automatically."
        )
        send_email(student_email, "StudyDesk Payment Not Received — Action Required", reject_msg)
        send_student_telegram(student_email, student_phone, f"<b>Payment Not Received</b>\n{reject_msg}")
        return {"message": "Rejected & payment-required message sent to student.", "status": "REJECTED"}

    otp = str(secrets.randbelow(900000) + 100000)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    supabase.table("payment_requests").update(
        {
            "status": "APPROVED",
            "otp_code": otp,
            "otp_expires_at": expires_at,
        }
    ).eq("id", request_id).execute()

    movies_line = ""
    if MOVIES_SITE_URL and MOVIES_SITE_PASSWORD:
        movies_line = f"\n\nPro Bonus — Free Movies Website:\nLink: {MOVIES_SITE_URL}\nPassword: {MOVIES_SITE_PASSWORD}"

    approve_msg = (
        f"Your StudyDesk payment of ₹{amount} has been approved!\n\n"
        f"Your 6-digit Pro Activation OTP is: {otp}\n"
        f"Enter this OTP on StudyDesk to unlock Pro immediately (valid for 24 hours).{movies_line}"
    )
    send_email(student_email, f"StudyDesk Pro Approved — Your OTP: {otp}", approve_msg)
    send_student_telegram(
        student_email,
        student_phone,
        f"<b>StudyDesk Pro Approved!</b>\n6-Digit OTP: <code>{otp}</code>\n\nEnter it on StudyDesk to unlock Pro.",
    )
    return {"message": "Approved! OTP sent automatically to student.", "otp_code": otp, "status": "APPROVED"}


@app.post("/api/payment/submit")
def submit_payment(data: PaymentSubmit, request: Request, bg: BackgroundTasks):
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"pay:{ip}", max_attempts=30, window_seconds=300)

    email = data.email.lower().strip()
    amount = 1 if data.amount == 1 else 99
    plan_label = "7-Day Trial (₹1)" if amount == 1 else "30-Day Pro Plan (₹99)"

    try:
        exists = supabase.table("payment_requests").select("id").eq("utr", data.utr).execute()
        if exists.data:
            raise HTTPException(status_code=409, detail="This 12-digit UTR was already submitted.")

        ins = supabase.table("payment_requests").insert(
            {
                "email": email,
                "phone": data.phone,
                "utr": data.utr,
                "amount": amount,
                "status": "PENDING",
            }
        ).execute()
        req_id = ins.data[0]["id"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    buttons = {
        "inline_keyboard": [
            [
                {"text": "✅ Approve & Send OTP", "callback_data": f"pay_approve:{req_id}"},
                {"text": "❌ Reject (No Payment)", "callback_data": f"pay_reject:{req_id}"},
            ]
        ]
    }
    bg.add_task(
        send_telegram,
        f"<b>New Payment Claim — {plan_label}</b>\n"
        f"Name: {data.name}\nEmail: <code>{email}</code>\n"
        f"Mobile: <code>{data.phone}</code>\nUTR: <code>{data.utr}</code>\n\n"
        f"Check your UPI app and tap a button below:",
        None,
        buttons,
    )
    return {
        "message": f"Payment for {plan_label} submitted! Once approved, your 6-digit OTP will be sent automatically to {email}."
    }


@app.get("/api/payment/status")
def check_payment_status(email: str):
    clean = email.lower().strip()
    res = (
        supabase.table("payment_requests")
        .select("status,amount,utr,created_at")
        .eq("email", clean)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        return {"status": "NONE", "message": "No payment request found yet."}
    st = res.data[0]["status"]
    msgs = {
        "PENDING": "Waiting for payment verification...",
        "APPROVED": "Payment approved! Check your Gmail (or Telegram) for the 6-digit OTP.",
        "ACTIVATED": "Your Pro plan is already active!",
        "REJECTED": "You haven't done the payment yet. Please do the payment first, then submit your valid UTR to receive the OTP.",
    }
    return {"status": st, "message": msgs.get(st, st)}


@app.post("/api/payment/verify-otp")
def verify_payment_otp(data: PaymentOTPVerify, request: Request):
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"potp:{ip}", max_attempts=20, window_seconds=300)
    email = data.email.lower().strip()

    try:
        res = (
            supabase.table("payment_requests")
            .select("*")
            .eq("email", email)
            .eq("status", "APPROVED")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not res.data:
        raise HTTPException(
            status_code=400,
            detail="No approved payment found for this email yet. Complete payment and wait for approval.",
        )

    row = res.data[0]
    if not secrets.compare_digest(row.get("otp_code") or "", data.otp_code):
        raise HTTPException(status_code=400, detail="Incorrect 6-digit OTP.")

    exp = datetime.fromisoformat(row["otp_expires_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > exp:
        raise HTTPException(status_code=400, detail="OTP has expired.")

    supabase.table("payment_requests").update({"status": "ACTIVATED", "otp_code": None}).eq("id", row["id"]).execute()

    days = 7 if row.get("amount") == 1 else 30
    now = datetime.now(timezone.utc)
    new_expiry = now + timedelta(days=days)
    supabase.table("subscriptions").upsert(
        {
            "email": email,
            "phone": row["phone"],
            "plan": "PRO",
            "active_until": new_expiry.isoformat(),
            "updated_at": now.isoformat(),
        },
        on_conflict="email",
    ).execute()

    return {
        "message": f"Pro Plan Activated ({days} Days)!",
        "active_until": new_expiry.strftime("%d %b %Y"),
        "movies_url": MOVIES_SITE_URL,
        "movies_password": MOVIES_SITE_PASSWORD,
    }


@app.post("/api/admin/login")
def admin_login(data: AdminLogin, request: Request):
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"login:{ip}", max_attempts=15, window_seconds=300)

    user_ok = secrets.compare_digest(data.username, ADMIN_USERNAME)
    try:
        pass_ok = password_hash.verify(data.password, ADMIN_PASSWORD_HASH)
    except Exception:
        pass_ok = False

    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Wrong username or password.")

    token = jwt.encode(
        {"sub": ADMIN_USERNAME, "role": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=8)},
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"access_token": token}


@app.get("/api/admin/requests")
def get_requests(_admin: str = Depends(verify_admin)):
    res = supabase.table("payment_requests").select("*").order("created_at", desc=True).limit(100).execute()
    return {"requests": res.data}


@app.post("/api/admin/decide")
def decide_request(data: AdminDecision, _admin: str = Depends(verify_admin)):
    return process_payment_decision(data.request_id, data.action)


# ==========================================================
# 2B. TELEGRAM BACKGROUND BOT WORKER (1-TAP ADMIN + STUDENT LINK)
# ==========================================================
async def telegram_bot_polling_loop():
    if not TELEGRAM_BOT_TOKEN:
        print("[Telegram Automation] Skipped (TELEGRAM_BOT_TOKEN not set in .env).", flush=True)
        return

    offset = 0
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    print("[Telegram Automation] Bot listener active — 1-tap Approve/Reject buttons ready.", flush=True)

    async with httpx.AsyncClient(timeout=35.0) as client:
        while True:
            try:
                r = await client.get(f"{base}/getUpdates", params={"timeout": 25, "offset": offset})
                if r.status_code != 200:
                    await asyncio.sleep(3)
                    continue
                updates = r.json().get("result") or []

                for upd in updates:
                    offset = upd["update_id"] + 1

                    # 1. Handle Admin 1-tap Approve / Reject button clicks
                    cb = upd.get("callback_query")
                    if cb:
                        from_chat = str(cb["message"]["chat"]["id"])
                        cb_data = cb.get("data", "")
                        if from_chat != str(TELEGRAM_CHAT_ID):
                            await client.post(
                                f"{base}/answerCallbackQuery",
                                json={"callback_query_id": cb["id"], "text": "Admin only!"},
                            )
                            continue

                        if cb_data.startswith(("pay_approve:", "pay_reject:")):
                            action = "approve" if cb_data.startswith("pay_approve:") else "reject"
                            req_id = cb_data.split(":", 1)[1]
                            try:
                                result = await asyncio.to_thread(process_payment_decision, req_id, action)
                                toast = result.get("message", "Done!")
                                if result.get("already_done"):
                                    tag = f"\n\nℹ️ Already processed ({result.get('status')})."
                                elif result.get("status") == "APPROVED":
                                    tag = f"\n\n✅ <b>APPROVED</b> — OTP <code>{result.get('otp_code')}</code> sent automatically!"
                                else:
                                    tag = "\n\n❌ <b>REJECTED</b> — Student notified to complete payment first."
                            except Exception as e:
                                toast = f"Error: {str(e)[:80]}"
                                tag = f"\n\n⚠️ {toast}"

                            await client.post(
                                f"{base}/answerCallbackQuery",
                                json={"callback_query_id": cb["id"], "text": toast[:180]},
                            )
                            orig_text = cb["message"].get("text", "")
                            await client.post(
                                f"{base}/editMessageText",
                                json={
                                    "chat_id": from_chat,
                                    "message_id": cb["message"]["message_id"],
                                    "text": orig_text + tag,
                                    "parse_mode": "HTML",
                                },
                            )
                        continue

                    # 2. Allow students to link their Telegram in 1 step for automatic OTPs
                    msg = upd.get("message")
                    if msg and msg.get("text"):
                        chat_id = str(msg["chat"]["id"])
                        text = msg["text"].strip()
                        if text.startswith("/start"):
                            parts = text.split(maxsplit=1)
                            if len(parts) == 2:
                                save_tg_user(parts[1], chat_id)
                                await client.post(
                                    f"{base}/sendMessage",
                                    json={
                                        "chat_id": chat_id,
                                        "text": "✅ Linked! StudyDesk will now send your Login & Payment OTPs here automatically.",
                                    },
                                )
                            else:
                                await client.post(
                                    f"{base}/sendMessage",
                                    json={
                                        "chat_id": chat_id,
                                        "text": "Welcome to StudyDesk Bot! Reply with your Gmail address or 10-digit mobile number to receive Login & Payment OTPs automatically here.",
                                    },
                                )
                        elif "@" in text or re.fullmatch(r"\d{10}", re.sub(r"\D", "", text)):
                            ident = text.lower() if "@" in text else re.sub(r"\D", "", text)[-10:]
                            save_tg_user(ident, chat_id)
                            await client.post(
                                f"{base}/sendMessage",
                                json={
                                    "chat_id": chat_id,
                                    "text": f"✅ Linked {ident}! You will now receive StudyDesk OTPs and payment updates automatically here.",
                                },
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Telegram Poller] Reconnecting: {e}", flush=True)
                await asyncio.sleep(4)


# ==========================================================
# 3. ALL 10 AI STUDY TOOLS (CONNECTED TO LAST 15 MESSAGES)
# ==========================================================

# TOOL 1: Photo-to-Doubt Solver (Multi-Step Follow-Ups Enabled)
@app.post("/api/tools/doubt-solver")
async def tool_doubt_solver(
    request: Request,
    question: str = Form(""),
    language: str = Form("Hinglish"),
    image: Optional[UploadFile] = File(None),
):
    sid = get_session_id(request)
    history = get_tool_history(sid, "doubt-solver")
    print(f"[Tool 1: Doubt Solver] sid={sid} | mem={len(history)} | has_image={bool(image and image.filename)}", flush=True)

    system_prompt = (
        f"You are a friendly top-rank Indian teacher for JEE, NEET, and CBSE Boards. "
        f"Solve clearly step-by-step in {language}. You have full memory of the last 15 messages in this doubt thread, "
        "so if the student asks a follow-up (like 'explain step 2 again' or 'what if mass is doubled?'), "
        "directly connect your answer to the earlier problem and solution.\n"
        "STRICT FORMATTING RULES:\n"
        "- Write in clean plain text ONLY.\n"
        "- Do NOT use LaTeX or Markdown symbols like $, $$, \\frac, \\times, \\sqrt, **, *, ###, or backticks.\n"
        "- Use normal readable math like (a)/(b), ×, ÷, ², √(x), =, and clear labels like Given:, Formula:, Step 1:, Final Answer:, and Exam Tip:."
    )

    user_q = question.strip()
    if image and image.filename:
        img_bytes = await image.read()
        prompt = user_q or "Read the question in this image carefully and solve it step-by-step."
        raw = await ask_groq_vision(
            prompt,
            [(img_bytes, image.content_type or "image/jpeg")],
            system_prompt=system_prompt,
            history=history,
        )
        saved_user = f"[Uploaded Question Photo] {user_q or 'Solve this photo question'}"
    else:
        if not user_q:
            raise HTTPException(status_code=400, detail="Upload a photo or type a question.")
        raw = await ask_groq(user_q, system_prompt=system_prompt, history=history)
        saved_user = user_q

    clean_ans = clean_readable_math(raw)
    append_tool_history(sid, "doubt-solver", saved_user, clean_ans)
    return {"answer": clean_ans, "memory_count": len(get_tool_history(sid, "doubt-solver"))}


# TOOL 2: Handwritten Notes to Flashcards & MCQs (With 15-Message Memory)
@app.post("/api/tools/flashcards")
async def tool_flashcards(
    request: Request,
    notes_text: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    sid = get_session_id(request)
    history = get_tool_history(sid, "flashcards")
    print(f"[Tool 2: Flashcards] sid={sid} | mem={len(history)}", flush=True)

    system_prompt = (
        "You are an exam revision generator. You remember the last 15 messages of notes and flashcards created with this student. "
        "If the student asks for more/harder cards from earlier notes, build on previous context without repeating old cards.\n"
        "Use clean plain text without LaTeX or markdown stars.\n"
        'Return ONLY valid JSON: {"flashcards":[{"front":"Short question or term","back":"Clear concise answer"}],'
        '"mcqs":[{"q":"Question text","options":["Option A","Option B","Option C","Option D"],"answer_index":0,"why":"1-line explanation"}]}'
    )

    notes_clean = notes_text.strip()
    if image and image.filename:
        img_bytes = await image.read()
        prompt = f"Create 6 flashcards and 4 MCQs from this notebook photo. Extra instructions: {notes_clean or 'None'}. Return ONLY valid JSON."
        raw = await ask_groq_vision(
            prompt,
            [(img_bytes, image.content_type or "image/jpeg")],
            system_prompt=system_prompt,
            history=history,
            json_mode=True,
        )
        saved_user = f"[Notebook Photo] {notes_clean or 'Create flashcards & MCQs'}"
    else:
        if not notes_clean:
            raise HTTPException(status_code=400, detail="Upload a notebook photo or paste notes first.")
        prompt = f"Create 6 flashcards and 4 exam-style MCQs based on this input (and prior notes if this is a follow-up):\n{notes_clean}\nReturn ONLY valid JSON."
        raw = await ask_groq(prompt, system_prompt=system_prompt, history=history, json_mode=True)
        saved_user = notes_clean[:1200]

    data = extract_json(raw, fallback={"flashcards": [], "mcqs": []})

    clean_cards = []
    for c in data.get("flashcards") or []:
        if isinstance(c, dict):
            front = clean_readable_math(str(c.get("front") or c.get("question") or c.get("q") or ""))
            back = clean_readable_math(str(c.get("back") or c.get("answer") or c.get("a") or ""))
            if front and back:
                clean_cards.append({"front": front, "back": back})

    clean_mcqs = []
    for m in data.get("mcqs") or []:
        if isinstance(m, dict):
            opts = [clean_readable_math(str(o)) for o in (m.get("options") or [])][:4]
            while len(opts) < 4:
                opts.append("None of the above")
            try:
                ans_idx = int(m.get("answer_index", 0))
            except Exception:
                ans_idx = 0
            ans_idx = max(0, min(ans_idx, len(opts) - 1))
            q_text = clean_readable_math(str(m.get("q") or m.get("question") or ""))
            if q_text:
                clean_mcqs.append(
                    {
                        "q": q_text,
                        "options": opts,
                        "answer_index": ans_idx,
                        "why": clean_readable_math(str(m.get("why") or m.get("explanation") or "Concept check.")),
                    }
                )

    if not clean_cards and not clean_mcqs:
        raise HTTPException(status_code=500, detail="Could not extract flashcards. Try clearer notes or text.")

    summary_mem = "Generated flashcards: " + "; ".join(c["front"] for c in clean_cards[:6])
    append_tool_history(sid, "flashcards", saved_user, summary_mem)
    return {"flashcards": clean_cards, "mcqs": clean_mcqs, "memory_count": len(get_tool_history(sid, "flashcards"))}


# TOOL 3: YouTube Lecture Summarizer (Supports Follow-Up Questions!)
class YtReq(BaseModel):
    url: str


@app.post("/api/tools/yt-summarizer")
async def tool_yt_summarizer(req: YtReq, request: Request):
    sid = get_session_id(request)
    history = get_tool_history(sid, "yt-summarizer")
    raw_input = req.url.strip()
    vid = try_extract_youtube_id(raw_input)

    system_prompt = (
        "You are a crisp Indian exam study assistant for YouTube lectures. "
        "You remember the last 15 messages of lecture summaries and follow-up questions with this student. "
        "Always reply in clean plain text without **, ###, or LaTeX $ symbols."
    )

    if not vid:
        if history and len(raw_input) >= 2:
            print(f"[Tool 3: YT Follow-Up] sid={sid} | mem={len(history)}", flush=True)
            ans = await ask_groq(raw_input, system_prompt=system_prompt, history=history)
            clean_ans = clean_readable_math(ans)
            append_tool_history(sid, "yt-summarizer", raw_input, clean_ans)
            return {"summary": clean_ans, "source": "memory-followup", "memory_count": len(get_tool_history(sid, "yt-summarizer"))}
        raise HTTPException(status_code=400, detail="Paste a valid YouTube video link (or ask a follow-up after summarizing a video).")

    print(f"[Tool 3: YT Summarizer] sid={sid} | video_id={vid} | mem={len(history)}", flush=True)
    base_instructions = (
        "Create a crisp 1-page exam revision cheat sheet from this YouTube lecture:\n"
        "1) 6-8 Key Concept Bullet Points\n"
        "2) Important Formulas / Definitions\n"
        "3) 3 High-Probability Exam Questions with short answers.\n"
    )

    cached = load_yt_cache().get(vid)
    if cached:
        summary = await ask_groq(base_instructions + "\nLecture Transcript:\n" + cached[:18000], system_prompt=system_prompt, history=history)
        clean_sum = clean_readable_math(summary)
        append_tool_history(sid, "yt-summarizer", f"Summarize YouTube video {vid}", clean_sum)
        return {"summary": clean_sum, "source": "cache"}

    transcript = await asyncio.to_thread(fetch_yt_transcript_lib_sync, vid)
    source = "youtube-captions"

    page_data = {}
    if not transcript or len(transcript) < 40:
        page_data = await asyncio.to_thread(fetch_yt_webpage_data_sync, vid)
        if page_data.get("transcript"):
            transcript = page_data["transcript"]
            source = "youtube-page-captions"

    if not transcript or len(transcript) < 40:
        transcript = await asyncio.to_thread(fetch_supadata_transcript_sync, vid)
        if transcript:
            source = "supadata"

    if not transcript or len(transcript) < 40:
        transcript = await asyncio.to_thread(transcribe_yt_via_groq_whisper_sync, vid)
        if transcript:
            source = "groq-whisper"

    if transcript and len(transcript.strip()) >= 40:
        save_yt_cache(vid, transcript)
        summary = await ask_groq(base_instructions + "\nLecture Transcript:\n" + transcript[:18000], system_prompt=system_prompt, history=history)
        clean_sum = clean_readable_math(summary)
        append_tool_history(sid, "yt-summarizer", f"Summarize YouTube video {vid}", clean_sum)
        return {"summary": clean_sum, "source": source}

    meta = await asyncio.to_thread(fetch_yt_oembed_sync, vid)
    title = meta.get("title") or page_data.get("title") or ""
    author = meta.get("author_name") or "YouTube Educator"
    desc = (page_data.get("description") or "")[:2000]
    if title:
        fallback_prompt = (
            f"A student wants an exam cheat sheet for the YouTube lecture titled: '{title}' by '{author}'.\n"
            f"Video Description/Chapters:\n{desc}\n\n"
            "Write a complete, structured revision cheat sheet covering:\n"
            "1) Core Concepts & Theory\n2) Essential Formulas / Equations\n3) 3 Practice Questions with short answers.\n"
            "Start with a 1-line note: (Summary built from lecture title & description because captions were unavailable)."
        )
        summary = await ask_groq(fallback_prompt, system_prompt=system_prompt, history=history)
        clean_sum = clean_readable_math(summary)
        append_tool_history(sid, "yt-summarizer", f"Summarize YouTube video '{title}'", clean_sum)
        return {"summary": clean_sum, "source": "topic-fallback"}

    raise HTTPException(status_code=400, detail="Could not read this YouTube link. Make sure the video is public.")


# TOOL 4: AI Mock Test Generator (Remembers Prior Tests So Questions Don't Repeat)
class MockTestReq(BaseModel):
    exam: str
    chapter: str
    difficulty: str
    count: int = 5


@app.post("/api/tools/mock-test")
async def tool_mock_test(req: MockTestReq, request: Request):
    sid = get_session_id(request)
    history = get_tool_history(sid, "mock-test")
    count = max(3, min(req.count, 15))
    print(f"[Tool 4: Mock Test] sid={sid} | mem={len(history)} | {req.exam} | {req.chapter}", flush=True)

    system_prompt = (
        "You are an Indian competitive exam mock test creator. You remember the last 15 messages of tests created for this student—"
        "do NOT repeat questions you already generated in earlier messages; create fresh, progressive questions.\n"
        "Use clean plain text without LaTeX $ or ** symbols.\n"
        'Return ONLY valid JSON: {"questions":[{"q":"Question text","options":["A","B","C","D"],"answer_index":0,"explanation":"Short solution"}]}'
    )
    user_prompt = f"Generate a {req.difficulty} level {req.exam} mock test with {count} fresh MCQs on chapter: {req.chapter}. Return ONLY valid JSON."
    raw = await ask_groq(user_prompt, system_prompt=system_prompt, history=history, json_mode=True)
    data = extract_json(raw, fallback={"questions": []})

    questions = []
    for item in data.get("questions") or []:
        if isinstance(item, dict):
            opts = [clean_readable_math(str(o)) for o in (item.get("options") or [])][:4]
            while len(opts) < 4:
                opts.append("None of these")
            try:
                idx = int(item.get("answer_index", 0))
            except Exception:
                idx = 0
            idx = max(0, min(idx, len(opts) - 1))
            q_text = clean_readable_math(str(item.get("q") or item.get("question") or ""))
            if q_text:
                questions.append(
                    {
                        "q": q_text,
                        "options": opts,
                        "answer_index": idx,
                        "explanation": clean_readable_math(str(item.get("explanation") or "")),
                    }
                )

    append_tool_history(
        sid,
        "mock-test",
        f"{req.exam} {req.difficulty} test on {req.chapter} ({count} Qs)",
        "Asked questions: " + " | ".join(q["q"][:90] for q in questions[:8]),
    )
    return {"questions": questions, "memory_count": len(get_tool_history(sid, "mock-test"))}


# TOOL 5: Long-Form 2-Host Study Podcast (Tony Stark & Logged-In Student, 5–20 Min)
def _split_text_for_acts(text: str, num_acts: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if num_acts <= 1 or len(clean) < 1200:
        return [clean[:10000]] * max(1, num_acts)
    chunk_size = max(1200, len(clean) // num_acts)
    chunks = []
    for i in range(num_acts):
        start = i * chunk_size
        end = len(clean) if i == num_acts - 1 else (i + 1) * chunk_size
        chunks.append(clean[start:end][:10000])
    return chunks


def _parse_dialogue_fallback(raw: str, student_name: str = "Student") -> list[dict]:
    parsed = extract_json(raw, fallback={})
    turns = []
    for d in parsed.get("dialogue") or []:
        if isinstance(d, dict) and str(d.get("text", "")).strip():
            raw_host = str(d.get("host", "A")).strip().lower()
            host = "B" if (raw_host.startswith(("b", "t")) or "tony" in raw_host or "stark" in raw_host) else "A"
            turns.append({"host": host, "text": clean_speech_text(str(d["text"]))})
    if turns:
        return turns

    safe_name = re.escape(student_name)
    pattern = rf"(?im)^(Tony\s*Stark|Tony|Stark|Host\s*B|Host\s*A|{safe_name})\s*:\s*(.+)$"
    for m in re.finditer(pattern, strip_thinking_tags(raw)):
        speaker = m.group(1).lower()
        host = "B" if ("tony" in speaker or "stark" in speaker or "b" in speaker) else "A"
        txt = clean_speech_text(m.group(2))
        if txt:
            turns.append({"host": host, "text": txt})
    return turns


async def _synthesize_long_edge_podcast(dialogue: list[dict], is_english: bool, out_path: Path) -> bool:
    """Synthesizes 5–20 min 2-host audio with distinct voices for Student (A) and Tony Stark (B)."""
    if is_english:
        voice_cfg = {
            "A": ("en-IN-PrabhatNeural", "+0%", "+2Hz"),
            "B": ("en-US-AndrewNeural", "+4%", "-4Hz"),
        }
        backup_voice = {"A": "en-IN-NeerjaNeural", "B": "en-US-GuyNeural"}
    else:
        voice_cfg = {
            "A": ("hi-IN-MadhurNeural", "+2%", "+7Hz"),
            "B": ("hi-IN-MadhurNeural", "+6%", "-9Hz"),
        }
        backup_voice = {"A": "hi-IN-SwaraNeural", "B": "hi-IN-MadhurNeural"}

    sem = asyncio.Semaphore(4)

    async def render_turn(idx: int, turn: dict) -> tuple[int, bytes]:
        line = clean_speech_text(turn.get("text", ""))[:1200]
        if not line:
            return idx, b""
        host = turn.get("host", "A")
        primary_voice, rate, pitch = voice_cfg.get(host, voice_cfg["A"])
        attempts = [
            (primary_voice, rate, pitch),
            (backup_voice.get(host, primary_voice), "+0%", "+0Hz"),
        ]

        async with sem:
            for voice, r_val, p_val in attempts:
                for _ in range(2):
                    try:
                        communicate = edge_tts.Communicate(line, voice, rate=r_val, pitch=p_val)
                        buf = bytearray()

                        async def _collect():
                            async for chunk in communicate.stream():
                                if chunk["type"] == "audio":
                                    buf.extend(chunk["data"])

                        await asyncio.wait_for(_collect(), timeout=25.0)
                        if len(buf) > 200:
                            return idx, bytes(buf)
                    except Exception as e:
                        print(f"[Podcast Edge-TTS] Turn {idx} ({voice}) retry: {e}", flush=True)
                        await asyncio.sleep(0.4)
            return idx, b""

    results = await asyncio.gather(*(render_turn(i, t) for i, t in enumerate(dialogue)))
    results.sort(key=lambda x: x[0])

    written_turns = 0
    with open(out_path, "wb") as fp:
        for _, audio_bytes in results:
            if audio_bytes:
                fp.write(audio_bytes)
                written_turns += 1

    return written_turns >= max(1, len(dialogue) // 2) and out_path.exists() and out_path.stat().st_size > 1000


def _gtts_long_save_sync(dialogue: list[dict], out_path: Path, lang: str = "en") -> bool:
    try:
        from gtts import gTTS

        with open(out_path, "wb") as fp:
            batch_text = ""
            for turn in dialogue:
                line = clean_speech_text(turn.get("text", ""))
                if len(batch_text) + len(line) > 2200 and batch_text:
                    gTTS(text=batch_text, lang=lang, tld="co.in" if lang == "en" else "com").write_to_fp(fp)
                    batch_text = line
                else:
                    batch_text = f"{batch_text} {line}".strip()
            if batch_text:
                gTTS(text=batch_text, lang=lang, tld="co.in" if lang == "en" else "com").write_to_fp(fp)
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception as e:
        print(f"[gTTS Long Fallback] Error: {e}", flush=True)
        return False


@app.post("/api/tools/pdf-podcast")
async def tool_pdf_podcast(
    request: Request,
    pdf: UploadFile = File(...),
    style: str = Form("Hinglish"),
    length_mode: str = Form("auto"),
    student_name: str = Form("Student"),
):
    sid = get_session_id(request)
    history = get_tool_history(sid, "pdf-podcast")
    host_student = re.sub(r"[^a-zA-Z0-9 .-]", "", (student_name or "Student").strip())[:35] or "Student"
    print(f"[Tool 5: Long Podcast] hosts=Tony Stark & {host_student} | file={pdf.filename} | mode={length_mode}", flush=True)

    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid .pdf file.")

    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    extracted = ""
    page_count = 0
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        extracted = "\n".join(page.get_text("text") for page in doc[:60]).strip()[:55000]

        if len(extracted) < 150:
            ocr_parts = []
            for start in range(0, min(page_count, 6), 3):
                imgs = [
                    (doc[p].get_pixmap(dpi=130).tobytes("jpeg", jpg_quality=80), "image/jpeg")
                    for p in range(start, min(start + 3, page_count))
                ]
                ocr_chunk = await ask_groq_vision(
                    "Extract all headings, explanations, definitions, and formulas from these study notes cleanly in plain text.",
                    imgs,
                )
                ocr_parts.append(ocr_chunk)
            extracted = "\n\n".join(ocr_parts).strip()
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read this PDF file. ({e})")

    if len(extracted) < 40:
        raise HTTPException(status_code=400, detail="Could not extract readable notes from this PDF.")

    mode = (length_mode or "auto").strip().lower()
    if mode == "15min":
        num_acts, turns_per_act = 5, 11
    elif mode == "5min":
        num_acts, turns_per_act = 2, 10
    else:
        if len(extracted) >= 14000 or page_count >= 11:
            num_acts, turns_per_act = 5, 11
        elif len(extracted) >= 5000 or page_count >= 5:
            num_acts, turns_per_act = 3, 11
        else:
            num_acts, turns_per_act = 2, 10

    is_english = style.strip().lower() == "english"
    chunks = _split_text_for_acts(extracted, num_acts)

    act_goals = []
    for idx in range(num_acts):
        if idx == 0:
            act_goals.append(
                f"ACT 1 (Opening & Hook): Start with awesome energy and a fun joke between {host_student} and Tony Stark. "
                f"{host_student} asks why this chapter feels confusing and asks the first big doubt; Tony Stark breaks it down with sharp humor and a cool real-world analogy."
            )
        elif idx == num_acts - 1:
            act_goals.append(
                f"ACT {num_acts} (Rapid-Fire Q&A, Exam Traps & Fun Outro): Cover the remaining notes, then Tony Stark quizzes {host_student} on 3 tricky exam questions "
                f"({host_student} answers playfully, falls for one classic exam trap, and Tony gives a genius memory shortcut). End with a quick recap and witty sign-off."
            )
        else:
            act_goals.append(
                f"ACT {idx + 1} (Deep Dive Q&A & Fun Banter): Continue smoothly with NO re-introductions. "
                f"{host_student} jumps in with curious 'Wait Tony, why does that happen?' questions and funny relatable comparisons; "
                "Tony Stark answers every doubt clearly with intuitive explanations, formulas spoken cleanly in words, and exam shortcuts."
            )

    lang_rules = (
        f"Language: Natural, witty English. {host_student} sounds like a sharp Indian student, and Tony Stark sounds like the charming, clever genius inventor mentoring them."
        if is_english
        else (
            f"Language: Natural, witty Indian Hinglish between {host_student} and Tony Stark. "
            "Write Hindi words in Devanagari script and keep English/science/exam terms in English "
            "(for example: 'अरे Tony, सच बताओ, ये formula याद कैसे रहेगा exam hall में?'). "
            "Keep the banter super fun, smart, and easy to follow."
        )
    )

    system_prompt = (
        "You are writing a real, entertaining 2-host study podcast starring:\n"
        f"- Host A ({host_student}): Curious, funny student co-host who asks real exam doubts, interrupts with 'Wait a second!' questions, and cracks relatable jokes.\n"
        f"- Host B (Tony Stark): Witty, brilliant mentor who calls {host_student} by name, jokes back, and explains every concept clearly using awesome analogies and exam hacks.\n"
        f"{lang_rules}\n"
        "STRICT RULES:\n"
        f"- Generate EXACTLY {turns_per_act} alternating turns (A, B, A, B...) in this Act.\n"
        "- DEPTH RULE: Every single turn MUST be 40 to 65 words (3 to 4 rich spoken sentences). Never write short 1-line turns!\n"
        "- Pure spoken words only: NO markdown, NO asterisks, NO emojis, NO stage directions like [laughs]. Write natural reactions in words.\n"
        'Return ONLY valid JSON: {"dialogue":[{"host":"A","text":"..."},{"host":"B","text":"..."}]}'
    )

    full_dialogue: list[dict] = []
    for idx in range(num_acts):
        print(f"[Tool 5: Long Podcast] Writing Act {idx + 1}/{num_acts}...", flush=True)
        prev_context = ""
        if full_dialogue:
            tail = "\n".join(
                f"{host_student if t['host'] == 'A' else 'Tony Stark'}: {t['text']}" for t in full_dialogue[-2:]
            )
            prev_context = f"\nPrevious lines just spoken (continue directly from here without greeting again):\n{tail}\n"

        act_prompt = (
            f"{act_goals[idx]}\n{prev_context}\n"
            f"Study Material for Act {idx + 1}/{num_acts}:\n{chunks[idx]}\n\n"
            f"Remember: EXACTLY {turns_per_act} rich alternating turns (40-65 words per turn) in valid JSON."
        )
        raw = await ask_groq(
            act_prompt,
            system_prompt=system_prompt,
            history=history[-4:] if idx == 0 else None,
            json_mode=True,
            max_completion_tokens=4096,
        )
        full_dialogue.extend(_parse_dialogue_fallback(raw, host_student))

    if not full_dialogue:
        raise HTTPException(status_code=500, detail="Could not generate podcast script. Please try again.")

    total_words = sum(len(t["text"].split()) for t in full_dialogue)
    est_minutes = max(1, round(total_words / 145))
    print(f"[Tool 5: Long Podcast] Script ready: {len(full_dialogue)} turns, {total_words} words (~{est_minutes} min). Recording...", flush=True)

    file_id = uuid.uuid4().hex[:10]
    mp3_path = AUDIO_DIR / f"podcast_{file_id}.mp3"
    audio_url = ""
    engine_used = "none"

    if await _synthesize_long_edge_podcast(full_dialogue, is_english, mp3_path):
        audio_url, engine_used = f"/audio/{mp3_path.name}", "edge-2host-long"
    elif await asyncio.to_thread(_gtts_long_save_sync, full_dialogue, mp3_path, "en" if is_english else "hi"):
        audio_url, engine_used = f"/audio/{mp3_path.name}", "gtts-long"

    readable_script = "\n\n".join(
        f"{host_student if t['host'] == 'A' else 'Tony Stark'}: {t['text']}" for t in full_dialogue
    )
    append_tool_history(
        sid,
        "pdf-podcast",
        f"PDF Podcast ({style}, ~{est_minutes} min, Tony Stark & {host_student}): {pdf.filename}",
        readable_script[:2400],
    )
    print(f"[Tool 5: Long Podcast] Done! engine={engine_used} | ~{est_minutes} min", flush=True)

    return {
        "script": readable_script,
        "audio_url": audio_url,
        "engine": engine_used,
        "turns": len(full_dialogue),
        "words": total_words,
        "estimated_minutes": est_minutes,
    }


# TOOL 6: Essay & Board Answer Grader (Tracks Revisions Across Drafts)
class EssayReq(BaseModel):
    topic: str
    essay: str
    max_marks: int = 10


@app.post("/api/tools/essay-grader")
async def tool_essay_grader(req: EssayReq, request: Request):
    if not req.essay.strip():
        raise HTTPException(status_code=400, detail="Paste your essay or answer first.")
    sid = get_session_id(request)
    history = get_tool_history(sid, "essay-grader")
    print(f"[Tool 6: Essay Grader] sid={sid} | mem={len(history)} | topic={req.topic}", flush=True)

    system_prompt = (
        "You are a CBSE/Board examiner who remembers the student's previous essay drafts from the last 15 messages. "
        "If the student submits a revised draft or asks a follow-up question, compare it to earlier drafts and note improvements.\n"
        "Write in clean plain text (no ** or ### symbols) with:\n"
        "1) Predicted Score (X/Y)\n2) Grammar & Clarity Fixes\n"
        "3) Missing Keywords/Points\n4) Improved High-Scoring Version."
    )
    user_prompt = f"Grade this answer out of {req.max_marks} marks.\nTopic: {req.topic or 'General'}\nAnswer:\n{req.essay}"
    feedback = clean_readable_math(await ask_groq(user_prompt, system_prompt=system_prompt, history=history))
    append_tool_history(sid, "essay-grader", user_prompt, feedback)
    return {"feedback": feedback, "memory_count": len(get_tool_history(sid, "essay-grader"))}


# TOOL 7: Formula & Concept Mind-Map Maker (Remembers Earlier Maps for Expansion)
class MindMapReq(BaseModel):
    topic: str


@app.post("/api/tools/mind-map")
async def tool_mind_map(req: MindMapReq, request: Request):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Enter a chapter or topic name.")
    sid = get_session_id(request)
    history = get_tool_history(sid, "mind-map")
    print(f"[Tool 7: Mind Map] sid={sid} | mem={len(history)} | topic={req.topic}", flush=True)

    system_prompt = (
        "You create clean Mermaid.js mindmaps for exam revision. You remember the student's previous mindmaps in the last 15 messages, "
        "so if they ask to expand a branch or add formulas to the earlier topic, update the mindmap accordingly.\n"
        "Return ONLY raw Mermaid code starting with `mindmap` on line 1. "
        "Indent with 2 spaces per level. Do NOT use code fences, quotes, colons, parentheses, or brackets inside branch labels."
    )
    code = strip_thinking_tags(await ask_groq(f"Mindmap request: {req.topic}", system_prompt=system_prompt, history=history, fast=True))
    code = code.replace("```mermaid", "").replace("```", "").strip()
    if not code.startswith("mindmap"):
        safe_topic = re.sub(r"[^a-zA-Z0-9 \-_]", "", req.topic.strip()) or "Topic"
        code = f"mindmap\n  root(({safe_topic}))\n    Core Concepts\n    Key Formulas\n    Exam Tips"

    append_tool_history(sid, "mind-map", req.topic.strip(), code)
    return {"mermaid": code, "memory_count": len(get_tool_history(sid, "mind-map"))}


# TOOL 8: Spoken English & Interview Coach (Full 15-Message Continuous Memory!)
class EnglishReq(BaseModel):
    user_speech: str = ""
    scenario: str = "Daily Conversation"


@app.post("/api/tools/english-buddy")
async def tool_english_buddy(req: EnglishReq, request: Request):
    sid = get_session_id(request)
    speech = (req.user_speech or "").strip()
    scenario = (req.scenario or "Daily Conversation").strip()
    tool_key = f"english-buddy:{scenario}"
    history = get_tool_history(sid, tool_key)
    print(f"[Tool 8: English Buddy] Step 1: sid={sid} | scenario={scenario} | mem={len(history)}...", flush=True)

    system_prompt = (
        f"You are a warm, realistic Spoken English & Interview Coach running a continuous multi-turn session. "
        f"Current Scenario: {scenario}.\n"
        "CRITICAL MEMORY RULES:\n"
        "- Carefully read all prior messages in this conversation (up to the last 15 messages).\n"
        "- Remember the student's name, background, answers, goals, and what you already asked.\n"
        "- NEVER restart the conversation or repeat greetings/questions you already asked.\n"
        "- Respond naturally to their latest reply in 2 short spoken sentences, ending with 1 relevant follow-up question.\n"
        'Return ONLY valid JSON: {"reply":"Natural 2-sentence reply with follow-up question",'
        '"better_way":"How a fluent speaker would phrase the student\'s latest sentence (or keep short encouragement if already fluent)",'
        '"tip":"1 short, practical speaking or interview tip"}'
    )

    if not speech:
        if history:
            user_prompt = (
                "The student paused and asked you to continue the session. "
                "Ask the next logical follow-up question based on our previous messages. Return ONLY valid JSON."
            )
        else:
            user_prompt = (
                f"Start our {scenario} session now. Greet the student warmly in 1 sentence and ask an engaging opening question. "
                "Return ONLY valid JSON."
            )
    else:
        user_prompt = f'Student said: "{speech}"\nContinue our {scenario} conversation naturally based on our history. Return ONLY valid JSON.'

    raw = await ask_groq(
        user_prompt,
        system_prompt=system_prompt,
        history=history,
        json_mode=True,
        fast=True,
        max_completion_tokens=1024,
    )
    print("[Tool 8: English Buddy] Step 2: Groq done! Generating voice...", flush=True)

    data = extract_json(
        raw,
        fallback={
            "reply": clean_readable_math(raw) or "That sounds great! Tell me a little more about that.",
            "better_way": speech or "Start by answering in two clear sentences.",
            "tip": "Keep sentences crisp and pause briefly between ideas.",
        },
    )

    reply_text = clean_readable_math(str(data.get("reply") or "Great point! Tell me more about that."))
    better_way = clean_readable_math(str(data.get("better_way") or speech or "Your sentence looks good!"))
    tip = clean_readable_math(str(data.get("tip") or "Maintain a steady pace and clear pronunciation."))

    append_tool_history(sid, tool_key, speech or "[Started / Continued session]", reply_text)
    updated_history = get_tool_history(sid, tool_key)

    audio_url = await generate_short_tts(reply_text, prefix="eng", lang="en")
    print(f"[Tool 8: English Buddy] Step 3: Done! mem={len(updated_history)} | audio={audio_url or 'text-only'}", flush=True)

    return {
        "reply": reply_text,
        "better_way": better_way,
        "tip": tip,
        "audio_url": audio_url,
        "history": updated_history,
        "memory_count": len(updated_history),
    }


# TOOL 9: Coding Doubt & Bug Fixer (Remembers Earlier Code & Follow-Up Errors)
class CodeReq(BaseModel):
    language: str
    code: str
    error: str = ""


@app.post("/api/tools/code-fixer")
async def tool_code_fixer(req: CodeReq, request: Request):
    if not req.code.strip() and not req.error.strip():
        raise HTTPException(status_code=400, detail="Paste your code or follow-up error first.")
    sid = get_session_id(request)
    history = get_tool_history(sid, "code-fixer")
    print(f"[Tool 9: Code Fixer] sid={sid} | mem={len(history)} | lang={req.language}", flush=True)

    system_prompt = (
        f"You are a patient {req.language} coding mentor. You remember the last 15 messages of code and errors discussed with this student, "
        "so if they share a follow-up error or modification request, build directly on the earlier code.\n"
        "Give:\n1) Simple explanation of the bug\n2) Complete corrected code\n3) Quick dry-run tip."
    )
    user_prompt = f"Language: {req.language}\nError: {req.error or 'None provided'}\nCode / Follow-up:\n{req.code}"
    solution = strip_thinking_tags(await ask_groq(user_prompt, system_prompt=system_prompt, history=history))
    append_tool_history(sid, "code-fixer", user_prompt, solution)
    return {"solution": solution, "memory_count": len(get_tool_history(sid, "code-fixer"))}


# TOOL 10: AI Daily Study Timetable Builder (Remembers Previous Schedule for Tweaks)
class TimetableReq(BaseModel):
    exam: str
    weak_chapters: str
    hours_per_day: int
    exam_date: str


@app.post("/api/tools/ai-timetable")
async def tool_ai_timetable(req: TimetableReq, request: Request):
    sid = get_session_id(request)
    history = get_tool_history(sid, "ai-timetable")
    print(f"[Tool 10: Timetable] sid={sid} | mem={len(history)} | exam={req.exam}", flush=True)

    system_prompt = (
        "You are an Indian exam study planner. You remember the student's previous timetable from the last 15 messages, "
        "so if they adjust hours or chapters, refine their plan seamlessly.\n"
        'Return ONLY valid JSON: {"slots":[{"time":"6:00 AM - 7:30 AM","subject":"Physics","task":"Revise weak topic + 15 PYQs"}]}'
    )
    user_prompt = (
        f"Build/update a realistic daily study timetable for {req.exam}, {req.hours_per_day} hours/day until {req.exam_date}. "
        f"Weak chapters / adjustments: {req.weak_chapters}. Return ONLY valid JSON."
    )
    raw = await ask_groq(user_prompt, system_prompt=system_prompt, history=history, json_mode=True, fast=True)
    data = extract_json(raw, fallback={"slots": [{"time": "Study Block", "subject": req.exam, "task": clean_readable_math(raw)}]})
    slots = [
        {
            "time": clean_readable_math(str(s.get("time", "Study Slot"))),
            "subject": clean_readable_math(str(s.get("subject", req.exam))),
            "task": clean_readable_math(str(s.get("task", ""))),
        }
        for s in (data.get("slots") or [])
        if isinstance(s, dict)
    ]

    summary_mem = "; ".join(f"{s['time']} {s['subject']}: {s['task']}" for s in slots[:8])
    append_tool_history(sid, "ai-timetable", user_prompt, summary_mem)
    return {"slots": slots, "memory_count": len(get_tool_history(sid, "ai-timetable"))}


# ==========================================================
# 3B. PYQ ARENA, 10-15 DPPs, DAILY PUZZLE, 3D LAB & JEE PDF NOTES
# ==========================================================
class PyqReq(BaseModel):
    exam: str = "JEE Main"
    student_class: str = "Class 11"
    subject: str = "Physics"
    chapter: str = "Laws of Motion"
    count: int = 5


@app.post("/api/tools/pyq-test")
async def tool_pyq_test(req: PyqReq, request: Request):
    sid = get_session_id(request)
    history = get_tool_history(sid, "pyq-test")
    count = max(3, min(req.count, 10))
    print(f"[PYQ Arena] {req.exam} | {req.student_class} | {req.subject} | {req.chapter}", flush=True)

    system_prompt = (
        f"You are an expert Indian {req.exam} PYQ database curator. "
        "Provide authentic, real past-year exam questions (PYQs) from recent years (2019-2026) for the requested chapter. "
        "Include the exact exam name, year, exam date, and shift (e.g., '27 Jan 2024 • Shift 1' for JEE Main, '26 May 2024 • Paper 1' for JEE Advanced, or '5 May 2024' for NEET), "
        "plus an estimated repeat-probability percentage (e.g., 84) based on how frequently that concept appears. "
        "Do NOT repeat questions already asked in recent conversation history. "
        "Use clean plain text math ONLY (no LaTeX $, \\frac, or ** symbols).\n"
        'Return ONLY valid JSON: {"questions":[{"q":"Full question text","options":["A","B","C","D"],"answer_index":0,'
        '"exam":"JEE Main","year":"2024","date":"27 Jan","shift":"Shift 1","chance_pct":85,'
        '"concept":"Core concept tested","explanation":"Clean step-by-step solution"}]}'
    )
    user_prompt = (
        f"Give {count} real {req.exam} PYQs for {req.student_class} {req.subject} — Chapter: {req.chapter}. "
        "Make sure difficulty matches real exam standards. Return ONLY valid JSON."
    )
    raw = await ask_groq(user_prompt, system_prompt=system_prompt, history=history, json_mode=True, max_completion_tokens=4096)
    data = extract_json(raw, fallback={"questions": []})

    questions = []
    for item in data.get("questions") or []:
        if not isinstance(item, dict):
            continue
        opts = [clean_readable_math(str(o)) for o in (item.get("options") or [])][:4]
        while len(opts) < 4:
            opts.append("None of these")
        try:
            ans_idx = max(0, min(int(item.get("answer_index", 0)), 3))
        except Exception:
            ans_idx = 0
        try:
            chance = max(35, min(int(item.get("chance_pct", 78)), 98))
        except Exception:
            chance = 78
        q_text = clean_readable_math(str(item.get("q") or item.get("question") or ""))
        if q_text:
            questions.append(
                {
                    "q": q_text,
                    "options": opts,
                    "answer_index": ans_idx,
                    "exam": clean_readable_math(str(item.get("exam") or req.exam)),
                    "year": str(item.get("year") or "2024"),
                    "date": clean_readable_math(str(item.get("date") or "Jan Session")),
                    "shift": clean_readable_math(str(item.get("shift") or ("Paper 1" if "Adv" in req.exam else "Shift 1"))),
                    "chance_pct": chance,
                    "concept": clean_readable_math(str(item.get("concept") or req.chapter)),
                    "explanation": clean_readable_math(str(item.get("explanation") or "")),
                }
            )

    append_tool_history(sid, "pyq-test", f"{req.exam} PYQs: {req.chapter}", " | ".join(q["q"][:80] for q in questions[:6]))
    return {"questions": questions}


class DppReq(BaseModel):
    exam: str = "JEE"
    subject: str = "Physics"
    chapter: str = "Work, Energy and Power"
    difficulty: str = "Medium to High"
    count: int = 12


@app.post("/api/tools/dpp-generator")
async def tool_dpp_generator(req: DppReq, request: Request):
    sid = get_session_id(request)
    history = get_tool_history(sid, "dpp-generator")
    count = max(10, min(req.count, 15))
    print(f"[DPP Engine] {req.exam} | {req.subject} | {req.chapter} | n={count}", flush=True)

    system_prompt = (
        f"You are a senior Kota JEE/NEET/CBSE DPP problem setter. Create a Daily Practice Problem (DPP) sheet of EXACTLY {count} "
        f"{req.difficulty} difficulty problems modeled on verified standard sources (NCERT Exemplar, HC Verma, Cengage, N. Awasthi, MS Chouhan, PYQ patterns). "
        "Mix conceptual traps and multi-step numericals. Use clean plain text math ONLY (no $, \\frac, or **).\n"
        'Return ONLY valid JSON: {"problems":[{"q":"Problem statement","options":["A","B","C","D"],"answer_index":0,'
        '"difficulty":"Medium","source_style":"HCV / JEE Main Style","solution":"Step-by-step clean solution"}]}'
    )
    user_prompt = f"Generate {count} {req.difficulty} DPP problems for {req.exam} — {req.subject}: {req.chapter}. Return ONLY valid JSON."
    raw = await ask_groq(user_prompt, system_prompt=system_prompt, history=history, json_mode=True, max_completion_tokens=6000)
    data = extract_json(raw, fallback={"problems": []})

    problems = []
    for p in data.get("problems") or []:
        if not isinstance(p, dict):
            continue
        opts = [clean_readable_math(str(o)) for o in (p.get("options") or [])][:4]
        while len(opts) < 4:
            opts.append("None of these")
        try:
            idx = max(0, min(int(p.get("answer_index", 0)), 3))
        except Exception:
            idx = 0
        q_text = clean_readable_math(str(p.get("q") or p.get("question") or ""))
        if q_text:
            problems.append(
                {
                    "q": q_text,
                    "options": opts,
                    "answer_index": idx,
                    "difficulty": clean_readable_math(str(p.get("difficulty") or "Medium-High")),
                    "source_style": clean_readable_math(str(p.get("source_style") or f"{req.exam} Standard")),
                    "solution": clean_readable_math(str(p.get("solution") or p.get("explanation") or "")),
                }
            )

    append_tool_history(sid, "dpp-generator", f"DPP {req.exam} {req.chapter}", f"Generated {len(problems)} DPP questions.")
    return {"problems": problems}


class PuzzleReq(BaseModel):
    exam: str = "JEE"
    subject: str = "Physics"


@app.post("/api/tools/daily-puzzle")
async def tool_daily_puzzle(req: PuzzleReq, request: Request):
    today = datetime.now().strftime("%d %b %Y")
    prompt = (
        f"Create 1 clever, brain-teasing Daily Concept Puzzle ({today}) for {req.exam} {req.subject} students "
        "(similar to Logic Bloom's intuition puzzles where a real-world paradox tests a core NCERT/JEE law). "
        "Use clean plain text without LaTeX or **.\n"
        'Return ONLY valid JSON: {"title":"Catchy puzzle title","scenario":"Short intriguing real-world puzzle",'
        '"options":["A","B","C","D"],"answer_index":0,"hint":"1-line clue without giving away answer",'
        '"explanation":"Aha-moment explanation linking to exam formula/law"}'
    )
    raw = await ask_groq(prompt, json_mode=True, fast=True)
    data = extract_json(raw, fallback={})
    opts = [clean_readable_math(str(o)) for o in (data.get("options") or ["Option A", "Option B", "Option C", "Option D"])][:4]
    return {
        "date": today,
        "title": clean_readable_math(str(data.get("title") or "Daily Concept Challenge")),
        "scenario": clean_readable_math(str(data.get("scenario") or "Think carefully about the underlying physical law.")),
        "options": opts,
        "answer_index": max(0, min(int(data.get("answer_index", 0) or 0), len(opts) - 1)),
        "hint": clean_readable_math(str(data.get("hint") or "Check conservation laws carefully.")),
        "explanation": clean_readable_math(str(data.get("explanation") or "")),
    }


class Visual3DReq(BaseModel):
    topic: str
    subject: str = "Physics"


@app.post("/api/tools/visual-3d")
async def tool_visual_3d(req: Visual3DReq, request: Request):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Enter a topic to visualize in 3D.")
    sid = get_session_id(request)
    history = get_tool_history(sid, "visual-3d")
    print(f"[3D Visual Lab] subject={req.subject} | topic={req.topic}", flush=True)

    system_prompt = (
        "You are a Three.js 3D STEM educator building interactive WebGL visualizations for Indian JEE/NEET/Board students. "
        "You must output ONLY valid JSON with:\n"
        '1) "title": short clean title\n'
        '2) "explanation": crisp, symbol-free conceptual explanation (key formula + intuition + exam tip)\n'
        '3) "legend": array of 3-5 short color-key strings (e.g., "Cyan sphere: Central Carbon atom")\n'
        '4) "scene_js": pure JavaScript code that runs inside an existing Three.js r128 environment where '
        "`THREE`, `scene`, `camera`, `addLabel(text, x, y, z, color)` are ALREADY defined, and you may define `window.animateScene = function(t){ ... }` "
        "for smooth continuous 3D motion (where `t` is elapsed seconds).\n"
        "STRICT JS RULES for scene_js:\n"
        "- Use ONLY standard Three.js r128 primitives (SphereGeometry, CylinderGeometry, BoxGeometry, TorusGeometry, ConeGeometry, ArrowHelper, BufferGeometry + Line, MeshStandardMaterial, MeshBasicMaterial, Group).\n"
        "- NEVER use external textures, fonts, TextGeometry, fetch, or import statements.\n"
        "- Position camera nicely if needed (default is 0, 4, 10) and keep objects centered near (0, 0, 0) within radius 5.\n"
        "- Add 3 to 5 clear calls to `addLabel('Label', x, y, z, '#38bdf8')` to name key parts in 3D space.\n"
        "- Make sure the JS has zero syntax errors.\n"
        'Return ONLY valid JSON: {"title":"...","explanation":"...","legend":["..."],"scene_js":"..."}'
    )
    user_prompt = f"Create a rich, animated interactive 3D diagram for {req.subject} topic: {req.topic}. Return ONLY valid JSON."
    raw = await ask_groq(user_prompt, system_prompt=system_prompt, history=history, json_mode=True, max_completion_tokens=4096)
    data = extract_json(raw, fallback={})

    scene_js = str(data.get("scene_js") or "").strip()
    scene_js = re.sub(r"^```(?:javascript|js)?\s*", "", scene_js)
    scene_js = re.sub(r"\s*```$", "", scene_js).strip()

    explanation = clean_readable_math(str(data.get("explanation") or f"3D visualization of {req.topic}."))
    title = clean_readable_math(str(data.get("title") or req.topic))
    legend = [clean_readable_math(str(x)) for x in (data.get("legend") or [])][:6]

    append_tool_history(sid, "visual-3d", f"3D Visual: {req.subject} - {req.topic}", explanation[:1200])
    return {"title": title, "explanation": explanation, "legend": legend, "scene_js": scene_js}


class NotesSheetReq(BaseModel):
    student_class: str
    subject: str
    chapter: str


@app.post("/api/tools/jee-notes-sheet")
async def tool_jee_notes_sheet(req: NotesSheetReq):
    prompt = (
        f"Create a complete, high-yield JEE/NEET/Board revision notes sheet for {req.student_class} {req.subject} — {req.chapter}.\n"
        "Use clean plain text math ONLY (no $, $$, \\frac, or **).\n"
        "Include:\n"
        "1) Core Concepts & Definitions\n"
        "2) Complete Formula Sheet & Key Reactions/Exceptions\n"
        "3) Top 5 Exam Traps & Memory Hacks\n"
        "4) 3 Solved High-Yield Numericals/Conceptual Examples."
    )
    notes = clean_readable_math(await ask_groq(prompt, max_completion_tokens=4096))
    return {"notes": notes}


# ==========================================================
# 4. NOTES PDF LIBRARY & LIVE WEBSOCKET CHAT
# ==========================================================
@app.get("/api/notes/list")
def list_notes():
    return {"notes": load_notes_db()}


@app.post("/api/notes/upload")
async def upload_note(
    title: str = Form(...),
    student_class: str = Form(...),
    subject: str = Form(...),
    uploader: str = Form("Student"),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    safe_name = f"{uuid.uuid4().hex[:8]}_{re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)}"
    (UPLOAD_DIR / safe_name).write_bytes(await file.read())
    items = load_notes_db()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "title": title.strip(),
        "student_class": student_class,
        "subject": subject,
        "uploader": uploader.strip() or "Student",
        "url": f"/uploads/{safe_name}",
        "created_at": datetime.now().strftime("%d %b %Y"),
    }
    items.insert(0, entry)
    save_notes_db(items)
    return {"note": entry}


class ChatManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self.history: list[dict] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        await ws.send_json({"type": "history", "messages": self.history[-40:], "online": len(self.active)})
        await self.broadcast({"type": "online", "online": len(self.active)})

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict):
        for conn in list(self.active):
            try:
                await conn.send_json(payload)
            except Exception:
                self.disconnect(conn)


chat_mgr = ChatManager()


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await chat_mgr.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            text = str(data.get("text", "")).strip()[:500]
            if not text:
                continue
            msg = {
                "type": "msg",
                "user": str(data.get("user", "Student")).strip()[:30] or "Student",
                "room": str(data.get("room", "General")).strip()[:20],
                "text": text,
                "time": datetime.now().strftime("%I:%M %p"),
                "online": len(chat_mgr.active),
            }
            chat_mgr.history = (chat_mgr.history + [msg])[-100:]
            await chat_mgr.broadcast(msg)
    except WebSocketDisconnect:
        chat_mgr.disconnect(ws)
        await chat_mgr.broadcast({"type": "online", "online": len(chat_mgr.active)})