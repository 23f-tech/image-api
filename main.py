import base64
import binascii
import io
import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel


app = FastAPI(title="Image QA API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageQuestionRequest(BaseModel):
    image_base64: str
    question: str


def clean_answer(answer: str) -> str:
    answer = answer.strip()

    answer = answer.replace("```json", "")
    answer = answer.replace("```text", "")
    answer = answer.replace("```", "")
    answer = answer.replace("**", "")
    answer = answer.strip()

    answer = re.sub(
        r"^(the\s+answer\s+is|final\s+answer|answer|result)\s*[:\-]?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()

    answer = answer.strip('"').strip("'").strip()

    numeric = answer.replace(",", "").strip()
    numeric = re.sub(r"^[₹$€£]\s*", "", numeric)

    numeric = re.sub(
        r"\s*(rupees?|dollars?|euros?|pounds?|units?|percent|%)\s*$",
        "",
        numeric,
        flags=re.IGNORECASE,
    ).strip()

    if re.fullmatch(r"-?\d+(?:\.\d+)?", numeric):
        return numeric

    return answer


def decode_image(encoded_image: str):
    encoded_image = encoded_image.strip()

    if encoded_image.startswith("data:"):
        if "," not in encoded_image:
            raise ValueError("Invalid image data URL.")

        encoded_image = encoded_image.split(",", 1)[1]

    # Remove spaces, tabs and line breaks from base64.
    encoded_image = re.sub(r"\s+", "", encoded_image)

    try:
        image_bytes = base64.b64decode(encoded_image)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid base64 image.") from error

    if not image_bytes:
        raise ValueError("Decoded image is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        # Reopen after verify().
        image = Image.open(io.BytesIO(image_bytes))

    except Exception as error:
        raise ValueError("The supplied data is not a valid image.") from error

    image_format = (image.format or "PNG").upper()

    mime_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }

    mime_type = mime_types.get(image_format, "image/png")

    # Convert unsupported formats to PNG.
    if image_format not in mime_types:
        converted = io.BytesIO()
        image.convert("RGB").save(converted, format="PNG")
        image_bytes = converted.getvalue()
        mime_type = "image/png"

    return image_bytes, mime_type


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Image question-answering API is ready",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/answer-image")
def answer_image(data: ImageQuestionRequest):
    try:
        if not data.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty.",
            )

        image_bytes, mime_type = decode_image(data.image_base64)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing in Render environment variables."
            )

        client = genai.Client(api_key=api_key)

        prompt = f"""
Examine the supplied image carefully and answer the question.

Question:
{data.question}

Rules:
- Return only the final answer.
- Do not explain.
- Do not return a sentence.
- For numeric answers, return only the number.
- Do not include commas, currency symbols, units, or percentage signs.
- For receipts, use the requested final or grand total, not subtotal,
  tax, cash paid, or change.
- For charts, carefully read all labels and values.
- When a calculation is requested, verify the arithmetic.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=100,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        answer = clean_answer(response.text)

        if not answer:
            raise RuntimeError("Cleaned answer is empty.")

        return {"answer": str(answer)}

    except HTTPException:
        raise

    except ValueError as error:
        print("INPUT ERROR:", repr(error), flush=True)

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        # This exact error will appear in Render Logs.
        print(
            "ANSWER-IMAGE ERROR:",
            type(error).__name__,
            str(error),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Image processing failed.",
        )