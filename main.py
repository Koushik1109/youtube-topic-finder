import os
import time
import uuid
import re
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import yt_dlp
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyBvAQcg0-ctv7REo3BrFAJLFix6y8nGBmU"))

app = FastAPI()

# Request model
class AskRequest(BaseModel):
    video_url: str
    topic: str

# Response model
class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str

# Helper function: validate HH:MM:SS
def validate_timestamp(ts: str):
    pattern = r"^\d{2}:\d{2}:\d{2}$"
    if not re.match(pattern, ts):
        raise ValueError("Timestamp must be in HH:MM:SS format")
    return ts

@app.post("/ask", response_model=AskResponse)
def ask_video(req: AskRequest):

    unique_id = str(uuid.uuid4())
    audio_filename = f"{unique_id}.mp3"

    try:
        # -----------------------------
        # 1️⃣ Download audio only
        # -----------------------------
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_filename,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([req.video_url])

        # -----------------------------
        # 2️⃣ Upload to Gemini Files API
        # -----------------------------
        uploaded_file = genai.upload_file(audio_filename)

        # -----------------------------
        # 3️⃣ Wait until ACTIVE
        # -----------------------------
        while uploaded_file.state.name != "ACTIVE":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)

        # -----------------------------
        # 4️⃣ Ask Gemini (Structured Output)
        # -----------------------------
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        You are analyzing an audio file.

        Find the FIRST time the following topic is spoken:
        "{req.topic}"

        Return ONLY the timestamp in HH:MM:SS format.
        """

        response = model.generate_content(
            [uploaded_file, prompt],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "string",
                            "description": "HH:MM:SS format"
                        }
                    },
                    "required": ["timestamp"]
                }
            }
        )

        timestamp = response.candidates[0].content.parts[0].text
        timestamp = eval(timestamp)["timestamp"]  # parse JSON safely in production

        validate_timestamp(timestamp)

        return {
            "timestamp": timestamp,
            "video_url": req.video_url,
            "topic": req.topic
        }

    finally:
        # -----------------------------
        # 5️⃣ Cleanup
        # -----------------------------
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
