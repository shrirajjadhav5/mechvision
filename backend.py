"""
backend.py — LangChain + Claude vision extraction engine
"""
import json
import re
import streamlit as st
from anthropic import Anthropic


class MechAnalyzerBackend:
    """Uses Claude claude-sonnet-4-20250514 vision to parse mechanical diagrams."""

    SYSTEM_PROMPT = """You are a precision mechanical engineering analyst.
When given an engineering diagram image, extract ALL labeled components and return
ONLY a valid JSON object (no markdown, no explanation).

The JSON must have this exact structure:
{
  "diagram_title": "string",
  "diagram_type": "string (e.g. Valve System, P&ID, Hydraulic, Pneumatic, Electrical)",
  "components": [
    {
      "ref_number": "string or int (the label number on the diagram)",
      "name": "string (descriptive component name)",
      "subsystem": "one of: Control | Mechanical | Fluid | Safety | Power | Structural | Sensor",
      "connection_count": int (count of visible connections/lines to this component),
      "complexity_score": float (1.0-5.0, your engineering judgment),
      "est_size_cm": float (estimated real-world size in cm, use engineering context),
      "is_actuated": int (1 if requires power/pneumatic/hydraulic actuation, else 0),
      "label_area_pct": float (approx % of diagram area this component occupies, 0.1-10.0),
      "subsystem_id": int (1=Control,2=Mechanical,3=Fluid,4=Safety,5=Power,6=Structural,7=Sensor)
    }
  ],
  "summary": "2-sentence description of what this system does"
}

Rules:
- Include EVERY labeled component (every numbered reference)
- Be precise with complexity_score: 1=simple bolt, 5=complex multi-function valve
- connection_count is the number of lines/pipes/wires visibly touching this component
- est_size_cm uses real engineering knowledge (a ball valve is ~15cm, a bolt is ~3cm)
- Return ONLY the JSON, nothing else"""

    def __init__(self):
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
        self.client = Anthropic(api_key=api_key) if api_key else None

    def extract_from_image(self, b64_image: str, mime_type: str) -> dict | None:
        if not self.client:
            # Return demo data so the app is testable without a key
            return self._demo_data()

        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": "Analyze this mechanical diagram and return the JSON."},
                    ],
                }],
            )
            raw = resp.content[0].text.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            st.error(f"Claude API error: {e}")
            return self._demo_data()

    # ── Demo data (used when no API key is present) ───────────────────────────
    @staticmethod
    def _demo_data() -> dict:
        return {
            "diagram_title": "Valve Patent Design — Fig. 2",
            "diagram_type": "Hydraulic / Pneumatic Valve System",
            "summary": (
                "A power take-off driven hydraulic valve system with emergency shutdown controls "
                "and a remote control box for managing fluid flow between a pressurised source "
                "and a receiving tank."
            ),
            "components": [
                {"ref_number": "50",  "name": "Valve body (upper)",    "subsystem": "Mechanical", "subsystem_id": 2, "connection_count": 4, "complexity_score": 4.2, "est_size_cm": 18, "is_actuated": 1, "label_area_pct": 3.1},
                {"ref_number": "52",  "name": "Actuator cap",          "subsystem": "Mechanical", "subsystem_id": 2, "connection_count": 2, "complexity_score": 2.8, "est_size_cm": 8,  "is_actuated": 1, "label_area_pct": 1.2},
                {"ref_number": "54",  "name": "Actuator top housing",  "subsystem": "Mechanical", "subsystem_id": 2, "connection_count": 3, "complexity_score": 3.0, "est_size_cm": 10, "is_actuated": 1, "label_area_pct": 1.4},
                {"ref_number": "56",  "name": "PTO drive unit",        "subsystem": "Power",      "subsystem_id": 5, "connection_count": 5, "complexity_score": 4.8, "est_size_cm": 20, "is_actuated": 1, "label_area_pct": 2.8},
                {"ref_number": "58",  "name": "Linkage arm",           "subsystem": "Mechanical", "subsystem_id": 2, "connection_count": 3, "complexity_score": 2.5, "est_size_cm": 6,  "is_actuated": 0, "label_area_pct": 0.8},
                {"ref_number": "60",  "name": "PTO label plate",       "subsystem": "Power",      "subsystem_id": 5, "connection_count": 2, "complexity_score": 1.5, "est_size_cm": 5,  "is_actuated": 0, "label_area_pct": 0.5},
                {"ref_number": "62",  "name": "Coupling body",         "subsystem": "Fluid",      "subsystem_id": 3, "connection_count": 4, "complexity_score": 3.5, "est_size_cm": 9,  "is_actuated": 0, "label_area_pct": 1.6},
                {"ref_number": "70",  "name": "Fluid connector A",     "subsystem": "Fluid",      "subsystem_id": 3, "connection_count": 3, "complexity_score": 2.8, "est_size_cm": 7,  "is_actuated": 0, "label_area_pct": 1.0},
                {"ref_number": "72",  "name": "Flexible hose",         "subsystem": "Fluid",      "subsystem_id": 3, "connection_count": 2, "complexity_score": 2.0, "est_size_cm": 25, "is_actuated": 0, "label_area_pct": 3.5},
                {"ref_number": "74",  "name": "Receiving tank",        "subsystem": "Fluid",      "subsystem_id": 3, "connection_count": 2, "complexity_score": 1.8, "est_size_cm": 22, "is_actuated": 0, "label_area_pct": 4.0},
                {"ref_number": "76",  "name": "End coupling A",        "subsystem": "Fluid",      "subsystem_id": 3, "connection_count": 3, "complexity_score": 2.5, "est_size_cm": 5,  "is_actuated": 0, "label_area_pct": 0.9},
                {"ref_number": "78",  "name": "End coupling B",        "subsystem": "Fluid",      "subsystem_id": 3, "connection_count": 3, "complexity_score": 2.5, "est_size_cm": 5,  "is_actuated": 0, "label_area_pct": 0.9},
                {"ref_number": "90",  "name": "Control box",           "subsystem": "Control",    "subsystem_id": 1, "connection_count": 6, "complexity_score": 4.5, "est_size_cm": 15, "is_actuated": 1, "label_area_pct": 6.0},
                {"ref_number": "94",  "name": "Pressure supply line",  "subsystem": "Safety",     "subsystem_id": 4, "connection_count": 2, "complexity_score": 1.5, "est_size_cm": 4,  "is_actuated": 0, "label_area_pct": 0.4},
                {"ref_number": "96",  "name": "Check valve",           "subsystem": "Safety",     "subsystem_id": 4, "connection_count": 3, "complexity_score": 2.2, "est_size_cm": 3,  "is_actuated": 0, "label_area_pct": 0.3},
                {"ref_number": "110", "name": "Air pressure input",    "subsystem": "Safety",     "subsystem_id": 4, "connection_count": 2, "complexity_score": 2.0, "est_size_cm": 6,  "is_actuated": 0, "label_area_pct": 0.7},
                {"ref_number": "114", "name": "Emergency stop fwd",    "subsystem": "Safety",     "subsystem_id": 4, "connection_count": 1, "complexity_score": 1.5, "est_size_cm": 4,  "is_actuated": 1, "label_area_pct": 2.0},
                {"ref_number": "116", "name": "Emergency stop aft",    "subsystem": "Safety",     "subsystem_id": 4, "connection_count": 1, "complexity_score": 1.5, "est_size_cm": 4,  "is_actuated": 1, "label_area_pct": 2.0},
            ],
        }
