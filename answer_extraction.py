import openai
from openai import OpenAI
import os
import json
openai.api_key = os.getenv("OPENAI_API_KEY")  # or directly assign the key here
import pydantic
import sys
from pydantic import BaseModel, Field
from typing import List
import asyncio
from  extraction import extract_multiple_pages_async
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client =OpenAI()

# Your question definitions
QUESTION_IDS = [
    "fullName", "address", "loanType", "loanAmount", "emiComfort", "monthlyObligations",
    "incomeSource", "salaryAmount", "designationEmployer", "otherIncomeSources",
    "businessRevenue", "businessProfit", "businessSalary",
    "atPropertyLocation", "propertyAddress", "propertyType", "propertyStructure",
    "propertyUsage", "landArea", "marketValue", "existingLoan", "existingLoanEmi"
]

# class ConversationItem(BaseModel):
#     id: str
#     role: str
#     message: str

# class TranscriptRequest(BaseModel):
#     conversations: List[ConversationItem]

def format_transcript(conversations):
    return "\n".join(f"{c['role'].capitalize()}: {c['message']}" for c in conversations)

@app.post("/summarize")
async def summarize_transcript(request: List[dict]):
    try:
        formatted_text = format_transcript(request)
        json_output = await extract_multiple_pages_async(
            input_type="TEXT",
            file_name="conversations",
            text_inputs=[formatted_text]
        )
        json_output.pop("page", None)

        return JSONResponse(content=json_output["response"][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    conversations = [
        {"id": "1", "role": "bot", "message": "First of all, what is your full name?"},
        {"id": "2", "role": "user", "message": "Hi, my name is Rajesh Kumar."},
        {"id": "3", "role": "bot", "message": "Where do you live? Please provide detailed address"},
        {"id": "4", "role": "user", "message": "I live in Mumbai, Maharashtra."},
        {"id": "5", "role": "bot", "message": "What type of loan do you want to take?"},
        {"id": "6", "role": "user", "message": "I need a Home Loan."},
        {"id": "7", "role": "bot", "message": "How much loan do you need? (Rs.)"},
        {"id": "8", "role": "user", "message": "The amount I need is 25 lakhs."},
        {"id": "9", "role": "bot", "message": "How much EMI can you pay every month? (Rs.)"},
        {"id": "10", "role": "user", "message": "I can afford an EMI of 25,000."},
        {"id": "11", "role": "bot", "message": "What are your current monthly obligations?"},
        {"id": "12", "role": "user", "message": "My current obligations are around 10,000."},
        {"id": "13", "role": "bot", "message": "What is your income source?"},
        {"id": "14", "role": "user", "message": "I am salaried and work at Infosys as a Senior Engineer."},
        {"id": "15", "role": "bot", "message": "What is your monthly salary?"},
        {"id": "16", "role": "user", "message": "My gross salary is 1,00,000 per month."},
        {"id": "17", "role": "bot", "message": "Do you have any other income sources?"},
        {"id": "18", "role": "user", "message": "Yes, I earn some rental income too."},
        {"id": "19", "role": "bot", "message": "Are you currently present at the collateral property location?"},
        {"id": "20", "role": "user", "message": "I am at the property now. It’s a residential flat at Andheri, Mumbai."},
        {"id": "21", "role": "bot", "message": "How is the property currently being used?"},
        {"id": "22", "role": "user", "message": "It is used as self-occupied and its market value is about 90 lakhs."},
        {"id": "23", "role": "bot", "message": "Is there any existing loan on the property?"},
        {"id": "24", "role": "user", "message": "No existing loan on it."}
    ]

    if "--run" in sys.argv:
        async def test_summary():
            formatted = format_transcript(conversations)
            result = await extract_multiple_pages_async(
                input_type="TEXT",
                file_name="test_convo",
                text_inputs=[formatted]
            )
            print(json.dumps(result["response"][0], indent=2))

        asyncio.run(test_summary())
    else:
        uvicorn.run("answer_extraction:app", host="localhost", port=4000, reload=True)