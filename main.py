import base64
import binascii
import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel


app = FastAPI(
    title="Multimodal Image Question Answering API",
    version="1.0.0"
)


# Allow the grader's Cloudflare Worker to call this API.
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


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured."
        )

    return genai.Client(api_key=api_key)


def clean_answer(answer: str) -> str:
    """
    Convert the model response into the exact short string
    expected by the automated grader.
    """

    answer = answer.strip()

    # Remove Markdown formatting.
    answer = answer.replace("```json", "")
    answer = answer.replace("```text", "")
    answer = answer.replace("```", "")
    answer = answer.replace("**", "")
    answer = answer.strip()

    # Remove common response prefixes.
    answer = re.sub(
        r"^(the\s+answer\s+is|final\s+answer|answer|result)\s*[:\-]\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()

    # Remove surrounding quotation marks.
    answer = answer.strip('"').strip("'").strip()

    # Prepare a possible numeric answer.
    numeric = answer.replace(",", "").strip()

    # Remove leading currency symbols.
    numeric = re.sub(r"^[₹$€£]\s*", "", numeric)

    # Remove common trailing words and symbols.
    numeric = re.sub(
        r"\s*(rupees?|dollars?|euros?|pounds?|units?|percent|%)\s*$",
        "",
        numeric,
        flags=re.IGNORECASE,
    ).strip()

    # Return only the number when the whole answer is numeric.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", numeric):
        return numeric

    return answer


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Image question-answering API is ready"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/answer-image")
def answer_image(data: ImageQuestionRequest):
    if not data.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    encoded_image = data.image_base64.strip()

    # Support both plain base64 and data URLs:
    # data:image/png;base64,AAAA...
    if encoded_image.startswith("data:"):
        if "," not in encoded_image:
            raise HTTPException(
                status_code=400,
                detail="Invalid image data URL."
            )

        encoded_image = encoded_image.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(
            encoded_image,
            validate=True
        )
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image."
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The decoded image is empty."
        )

    prompt = f"""
You are answering a question about an image for an automated grader.

Question:
{data.question}

Examine the entire image carefully.

Possible image categories include:
- receipt
- invoice
- pie chart
- bar chart
- table
- infographic

Rules:
1. Return only the final answer.
2. Do not explain your reasoning.
3. Do not write a sentence.
4. For numeric answers, return only the number.
5. Do not include commas in numeric answers.
6. Do not include currency symbols.
7. Do not include units.
8. Do not include a percentage symbol.
9. For a receipt, distinguish grand total from subtotal, tax, cash, and change.
10. For a bar chart, read every relevant bar and calculate carefully.
11. For a pie chart, use visible labels, values, percentages, and slice sizes.
12. Verify arithmetic before answering.
13. The response must be suitable for exact automated comparison.
"""

    try:
        client = get_client()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png"
                        ),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=100,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=500,
                detail="The model returned an empty answer."
            )

        answer = clean_answer(response.text)

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="The final answer was empty."
            )

        return {"answer": str(answer)}

    except HTTPException:
        raise

    except Exception as error:
        print("Gemini processing error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to process the image."
        )