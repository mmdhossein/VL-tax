import base64
import json
import re
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """
    Convert a local image file to a base64-encoded data URI string
    suitable for use in LangChain image_url messages.

    Args:
        image_path: Path to the image file (jpg, png, webp, gif)

    Returns:
        A data URI string like: "data:application/octet-stream;base64,/9j/4AAQ..."
    """
    path = Path(image_path)

    extension = path.suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }

    mime_type = mime_map.get(extension)
    if mime_type is None:
        raise ValueError(
            f"Unsupported image format '{extension}'. "
            f"Supported formats: {list(mime_map.keys())}"
        )

    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ---------------------------------------------------------------------------
# Examples for few shots
# ---------------------------------------------------------------------------
FEW_SHOT_AUTHENTIC  = image_to_base64("examples/authentic_card.jpg")
FEW_SHOT_FRAUD_SCREEN = image_to_base64("examples/fraud_phone_screen.jpg")
FEW_SHOT_FRAUD_EDITED = image_to_base64("examples/fraud_edited_code.jpg")
FEW_SHOT_FRAUD_PRINT  = image_to_base64("examples/fraud_printed_template.png")


def extract_national_code_with_langchain(image_data: str) -> str:
    """
    Extract and validate a national ID code from an Iranian national ID card image.

    Uses role-playing + few-shot prompting to detect fraud before extraction.
    Rejects screenshots, edited images, printed fakes, wrong document types, etc.

    Args:
        image_data: Base64-encoded image as a data URI
                    (use image_to_base64() to prepare it)

    Returns:
        The 10-digit national code string if authentic,
        or "NOT_FOUND" / "FRAUD_DETECTED" with a reason appended.
    """
    llm = ChatOpenAI(model="gpt-4o", max_tokens=512, temperature=0)

    # ------------------------------------------------------------------
    # SYSTEM — establishes the role and strict output contract
    # ------------------------------------------------------------------
    system_message = SystemMessage(content="""شما "سیستم تأیید هویت کارت ملی" هستید — یک سیستم تخصصی که وظیفه‌اش
تشخیص اصالت کارت‌های ملی ایرانی و استخراج کد ملی است.

نقش شما:
شما به‌عنوان یک کارشناس خبره جعل اسناد عمل می‌کنید که هزاران کارت ملی واقعی و جعلی
دیده‌اید. شما می‌دانید دقیقاً چه چیزی یک کارت را اصیل یا جعلی می‌کند.

بررسی‌های اجباری:
۱. آیا تصویر از صفحه‌نمایش گرفته شده؟ (خطوط موآره، پیکسل‌های درشت، انعکاس نور)
۲. آیا متن یا کد روی تصویر دیگری چاپ‌شده یا پیست شده؟ (لبه‌های ناهموار، سایه مصنوعی)
۳. آیا نوردهی و رنگ‌ها طبیعی هستند؟ (نور مصنوعی، رنگ‌آمیزی ناهمخوان)
۴. آیا هولوگرام یا المان‌های امنیتی قابل مشاهده است؟
۵. آیا فونت و قالب با استاندارد کارت ملی ایران مطابقت دارد؟
۶. آیا سند نوع دیگری است؟ (گواهینامه، گذرنامه، شناسنامه، کارت خارجی)
۷. آیا کارت منقضی یا باطل‌شده (مثلاً سوراخ‌شده) است؟

خروجی شما باید دقیقاً این فرمت JSON باشد و هیچ متن اضافه‌ای نداشته باشد:
{
  "is_authentic": true یا false,
  "fraud_indicators": ["دلیل ۱", "دلیل ۲"],
  "national_code": "۱۰ رقم یا NOT_FOUND",
  "confidence": عدد بین 0 تا 100
}""")

    # ------------------------------------------------------------------
    # FEW-SHOT TURN 1 — authentic card example
    # ------------------------------------------------------------------
    FEW_SHOT_AUTHENTIC_human = HumanMessage(content=[
        {
            "type": "text",
            "text": "این تصویر را بررسی کن و کد ملی را استخراج کن."
        },
        {
            "type": "image_url",
            "image_url": {"url": FEW_SHOT_AUTHENTIC}
        }
    ])
    FEW_SHOT_AUTHENTIC_ai = AIMessage(content="""{
  "is_authentic": true,
  "fraud_indicators": [],
  "national_code": "0012345678",
  "confidence": 96
}""")


    # ------------------------------------------------------------------
    # FEW-SHOT TURN 3 — fraud: photo of a phone screen
    # ------------------------------------------------------------------
    FEW_SHOT_FRAUD_SCREEN_human = HumanMessage(content=[
        {
            "type": "text",
            "text": "این تصویر را بررسی کن و کد ملی را استخراج کن."
        },
        {
            "type": "image_url",
            "image_url": {"url": FEW_SHOT_FRAUD_SCREEN}
        }
    ])
    FEW_SHOT_FRAUD_SCREEN_ai = AIMessage(content="""{
  "is_authentic": false,
  "fraud_indicators": [
    "تصویر از روی صفحه‌نمایش گرفته شده — خطوط موآره و انعکاس نور قابل مشاهده است",
    "هولوگرام و المان‌های امنیتی فیزیکی وجود ندارد"
  ],
  "national_code": "NOT_FOUND",
  "confidence": 15
}""")

    # ------------------------------------------------------------------
    # FEW-SHOT TURN 4 — fraud: digitally edited national code
    # ------------------------------------------------------------------
    few_shot_fraud_edited_human = HumanMessage(content=[
        {
            "type": "text",
            "text": "این تصویر را بررسی کن و کد ملی را استخراج کن."
        },
        {
            "type": "image_url",
            "image_url": {"url": FEW_SHOT_FRAUD_EDITED}
        }
    ])
    few_shot_fraud_edited_ai = AIMessage(content="""{
  "is_authentic": false,
  "fraud_indicators": [
    "کد ملی به‌صورت دیجیتال جایگزین شده — لبه‌های ناهموار و اختلاف رنگ زمینه مشهود است",
    "فونت کد ملی با سایر متن‌های کارت مطابقت ندارد"
  ],
  "national_code": "NOT_FOUND",
  "confidence": 10
}""")

    # ------------------------------------------------------------------
    # FEW-SHOT TURN 5 — fraud: printed template (no security features)
    # ------------------------------------------------------------------
    few_shot_fraud_print_human = HumanMessage(content=[
        {
            "type": "text",
            "text": "این تصویر را بررسی کن و کد ملی را استخراج کن."
        },
        {
            "type": "image_url",
            "image_url": {"url": FEW_SHOT_FRAUD_PRINT}
        }
    ])
    few_shot_fraud_print_ai = AIMessage(content="""{
  "is_authentic": false,
  "fraud_indicators": [
    "کارت چاپ‌شده روی کاغذ معمولی است — بافت کارت پلاستیکی وجود ندارد",
    "هیچ هولوگرام یا المان امنیتی دیده نمی‌شود",
    "کیفیت چاپ پایین و رنگ‌ها کم‌رنگ هستند"
  ],
  "national_code": "NOT_FOUND",
  "confidence": 5
}""")

    # ------------------------------------------------------------------
    # ACTUAL QUERY — the image under inspection
    # ------------------------------------------------------------------
    query_message = HumanMessage(content=[
        {
            "type": "text",
            "text": "این تصویر را بررسی کن و کد ملی را استخراج کن."
        },
        {
            "type": "image_url",
            "image_url": {"url": image_data}
        }
    ])

    # ------------------------------------------------------------------
    # Assemble the full conversation and invoke
    # ------------------------------------------------------------------
    messages = [
        system_message,
        FEW_SHOT_AUTHENTIC_human,  FEW_SHOT_AUTHENTIC_ai,
        FEW_SHOT_FRAUD_SCREEN_human, FEW_SHOT_FRAUD_SCREEN_ai,
        few_shot_fraud_edited_human, few_shot_fraud_edited_ai,
        few_shot_fraud_print_human,  few_shot_fraud_print_ai,
        query_message,
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # ------------------------------------------------------------------
    # Parse and validate the JSON response
    # ------------------------------------------------------------------
    try:
        # Strip markdown code fences if the model adds them anyway
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Unparseable response — treat as unverifiable
        return "NOT_FOUND: پاسخ مدل قابل پردازش نبود"

    is_authentic: bool  = result.get("is_authentic", False)
    confidence: int     = result.get("confidence", 0)
    national_code: str  = result.get("national_code", "NOT_FOUND")
    fraud_indicators    = result.get("fraud_indicators", [])

    # Reject if fraud was detected
    if not is_authentic:
        reasons = " | ".join(fraud_indicators) if fraud_indicators else "دلیل نامشخص"
        return f"FRAUD_DETECTED: {reasons}"

    # Reject if confidence is too low even on a "authentic" response
    if confidence < 80:
        return f"NOT_FOUND: اطمینان مدل پایین است ({confidence}%)"

    # Reject if the code itself is missing or malformed
    if not national_code or national_code == "NOT_FOUND":
        return "NOT_FOUND: کد ملی در تصویر یافت نشد"

    if not re.fullmatch(r"\d{10}", national_code):
        return f"NOT_FOUND: فرمت کد ملی نامعتبر است ({national_code})"

    return national_code
