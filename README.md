

# OCR Service Enhanced — National ID Verification

Hardened Iranian national ID card OCR using LangChain + GPT-4o with anti-forgery validation.


![diagram](img/AI-Powered_ID_Shield_System.png)

## Files

project-root/
├── ocr_service_enhanced.py   # Main module with both functions
└── examples/                 # Few-shot example images (required)
    ├── authentic_card.jpg
    ├── fraud_phone_screen.jpg
    ├── fraud_edited_code.jpg
    └── fraud_printed_template.jpg


## How It Works

Two functions are exposed:

**`image_to_base64(image_path)`**
Converts a local image file to a base64 data URI. Supports `.jpg`, `.png`, `.webp`, `.gif`.

**`extract_national_code_with_langchain(image_data)`**
Takes a base64 data URI and returns the 10-digit national code, or a rejection reason.

```python
from ocr_service_enhanced import image_to_base64, extract_national_code_with_langchain

image_data = image_to_base64("path/to/card.jpg")
result = extract_national_code_with_langchain(image_data)
print(result)
# "0012345678"          → success
# "FRAUD_DETECTED: ..." → fraud found
# "NOT_FOUND: ..."      → unreadable or low confidence
```

## Anti-Forgery Enhancements

### 1. Role-Playing System Prompt
The model is instructed to act as an expert document fraud examiner. It checks for:
- Screenshot / phone screen artifacts (moiré patterns, glare)
- Digitally overlaid or replaced text (edge artifacts, font mismatch)
- Printed paper templates (no hologram, low print quality)
- Abnormal lighting or color inconsistencies
- Wrong document type (passport, driver's license, birth certificate)
- Expired or hole-punched (cancelled) cards

### 2. Few-Shot Multimodal Examples
Five real conversation turns (Human → AI pairs) are prepended before the target image:

| Example file | Label | Teaches the model to detect |
|---|---|---|
| `authentic_card_1.jpg` | ✅ Authentic | Normal acceptance path |
| `authentic_card_2.jpg` | ✅ Authentic | Varied lighting, still valid |
| `fraud_phone_screen.jpg` | ❌ Fraud | Screen capture artifacts |
| `fraud_edited_code.jpg` | ❌ Fraud | Digitally replaced national code |
| `fraud_printed_template.jpg` | ❌ Fraud | Printed paper template |

### 3. Strict JSON Contract
The model must return only this structure:

```json
{
  "is_authentic": true,
  "fraud_indicators": [],
  "national_code": "0012345678",
  "confidence": 96
}
```

### 4. Post-Processing Validation Gates
The parsed response passes through four rejection gates before a code is returned:

| Gate | Condition | Return value |
|---|---|---|
| Fraud flag | `is_authentic` is `false` | `FRAUD_DETECTED: <reasons>` |
| Low confidence | `confidence < 80` | `NOT_FOUND: اطمینان مدل پایین است` |
| Missing code | `national_code` is empty or `NOT_FOUND` | `NOT_FOUND: کد ملی یافت نشد` |
| Bad format | Code is not exactly 10 digits | `NOT_FOUND: فرمت نامعتبر` |

## Setup

```bash
pip install langchain langchain-openai openai
export OPENAI_API_KEY="your-key-here"
```

Populate the `examples/` folder with real card images before running.
The five example files are loaded once at module import time.

## Requirements

- Python 3.9+
- `langchain-openai >= 0.1.0`
- GPT-4o access on your OpenAI API key
- Example images in `examples/` matching the filenames above