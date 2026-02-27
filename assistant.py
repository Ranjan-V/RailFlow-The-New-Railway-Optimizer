"""
assistant.py
LLM-powered controller assistant using Google Gemini API.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SYSTEM_PROMPT = """You are a Railway Section Controller Assistant for Indian Railways.
Help controllers make real-time decisions on the Hyderabad-Vijayawada-Chennai-Bangalore corridor.
Use Indian Railways terminology. Be direct and brief.
Priority: Rajdhani > Shatabdi > SF > Express > Passenger/MEMU.
Always end with: RECOMMENDATION: [one clear action]"""


def build_context(corridor_trains, optimization_result=None, disruptions=None):
    lines = ["=== SECTION STATUS ==="]
    lines.append(f"Active trains: {len(corridor_trains)}")

    if disruptions:
        lines.append("DISRUPTIONS:")
        for tn, delay in disruptions.items():
            lines.append(f"  Train {tn}: +{delay} min")

    # Only send 5 trains to keep tokens low
    lines.append("TRAINS (sample):")
    for t in corridor_trains[:5]:
        stops = t.get("stops", [])
        first = stops[0] if stops else {}
        last = stops[-1] if stops else {}
        lines.append(
            f"  [{t.get('type','?')}] {t['number']} {t.get('name','')} | "
            f"{first.get('station_code','?')}→{last.get('station_code','?')}"
        )

    if optimization_result and optimization_result.get("status") in ("OPTIMAL", "FEASIBLE"):
        lines.append(f"OPTIMIZATION: {optimization_result['status']} | delay={optimization_result['total_delay_minutes']}min")
        delayed = [t for t in optimization_result.get("trains", []) if t["total_delay"] > 0]
        for t in delayed[:3]:
            lines.append(f"  {t['number']}: +{t['total_delay']}min")

    return "\n".join(lines)


_chat_history = []


def ask_assistant(question, corridor_trains, optimization_result=None, disruptions=None, conversation_history=None):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-gemini-api-key-here":
        return {"answer": "GEMINI_API_KEY not set in .env file. Get free key at aistudio.google.com/app/apikey", "tokens_used": 0}

    try:
        client = genai.Client(api_key=api_key)
        context = build_context(corridor_trains, optimization_result, disruptions)
        full_question = f"{context}\n\nQUESTION: {question}"

        contents = []
        hist = conversation_history or _chat_history
        for msg in hist[-4:]:  # only last 2 exchanges to save tokens
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=full_question)]))

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=200,  # keep responses short
            ),
            contents=contents,
        )
        answer = response.text
        tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
        return {"answer": answer, "tokens_used": tokens}

    except Exception as e:
        return {"answer": f"Gemini API error: {str(e)}", "tokens_used": 0}


def reset_history():
    global _chat_history
    _chat_history = []


if __name__ == "__main__":
    from data_loader import build_corridor_data
    _, _, _, _, corridor_trains = build_corridor_data()

    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your-gemini-api-key-here":
        print("Add GEMINI_API_KEY to your .env file first!")
        print("Get free key: https://aistudio.google.com/app/apikey")
    else:
        result = ask_assistant(
            question="Train 47154 is 30 minutes late. Which trains are affected?",
            corridor_trains=corridor_trains[:5],
            disruptions={"47154": 30},
        )
        print("\nAssistant:")
        print(result["answer"])