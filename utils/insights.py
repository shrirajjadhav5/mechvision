"""
insights.py — Uses Claude to produce plain-language business insights
"""
import json
import re
import streamlit as st
from anthropic import Anthropic


class InsightsGenerator:

    SYSTEM = """You are a senior engineering management consultant who translates
machine learning results into clear, actionable business language for non-technical executives.

You will receive:
1. A dataset of mechanical components with their attributes
2. ML model results (accuracy, predictions, feature importance)

Return ONLY a valid JSON object (no markdown, no preamble) with this exact structure:
{
  "insights": [
    {
      "type": "risk | opportunity | observation",
      "title": "Short headline (max 8 words)",
      "body": "2-3 sentence plain explanation. No jargon. Include specific numbers from the data."
    }
  ],
  "exec_table": [
    {
      "Area": "string",
      "Finding": "string (1 sentence)",
      "Business Impact": "string (£/$ or % or risk level)",
      "Confidence": "High | Medium | Low"
    }
  ],
  "actions": [
    {
      "action": "Specific action (verb + object)",
      "reason": "1-sentence business justification",
      "priority": "High | Medium | Low"
    }
  ]
}

Rules:
- insights: exactly 4 items (2 risks, 1 opportunity, 1 observation)
- exec_table: exactly 5 rows covering: model accuracy, top risk component, cost driver, safety finding, recommendation
- actions: exactly 4 prioritised actions
- Use plain English only. No ML terminology in the body text.
- Mention actual component names, numbers, and percentages from the data."""

    def __init__(self):
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
        self.client = Anthropic(api_key=api_key) if api_key else None

    def generate(self, df, ml_results: dict, model_name: str, target: str) -> dict:
        summary = {
            "total_components": len(df),
            "subsystems": df["subsystem"].value_counts().to_dict() if "subsystem" in df.columns else {},
            "avg_complexity": round(df["complexity_score"].mean(), 2) if "complexity_score" in df.columns else None,
            "high_complexity": df[df["complexity_score"] >= 4]["name"].tolist() if "complexity_score" in df.columns else [],
            "actuated_count": int(df["is_actuated"].sum()) if "is_actuated" in df.columns else 0,
            "model": model_name,
            "target": target,
            "metrics": ml_results.get("metrics", {}),
            "top_features": ml_results.get("feature_importance", [])[:3],
            "worst_predictions": ml_results["predictions_df"].nlargest(3, "Error%")[["Component","Actual","Predicted","Error%"]].to_dict("records") if ml_results.get("predictions_df") is not None else [],
        }

        prompt = f"Here is the mechanical system analysis data:\n\n{json.dumps(summary, indent=2)}\n\nGenerate the business insights JSON."

        if not self.client:
            return self._demo_insights(summary)

        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=self.SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            st.warning(f"Insight generation fell back to demo: {e}")
            return self._demo_insights(summary)

    def to_report(self, df, ml_results: dict, insights: dict) -> str:
        lines = [
            "=" * 60,
            "MECHVISION AI — ENGINEERING INTELLIGENCE REPORT",
            "=" * 60,
            "",
            f"Total components analysed : {len(df)}",
            f"ML model used             : {ml_results.get('model_name','')}",
            f"Target variable           : {ml_results.get('target','')}",
            f"R² accuracy               : {ml_results['metrics'].get('R2', ml_results['metrics'].get('Accuracy',''))}",
            f"RMSE                      : {ml_results['metrics'].get('RMSE','')}",
            "",
            "─" * 60,
            "BUSINESS INSIGHTS",
            "─" * 60,
        ]
        for item in insights.get("insights", []):
            lines += ["", f"[{item.get('type','').upper()}] {item.get('title','')}", item.get("body","")]

        lines += ["", "─" * 60, "RECOMMENDED ACTIONS", "─" * 60]
        for i, a in enumerate(insights.get("actions", []), 1):
            lines += [f"{i}. [{a.get('priority','')}] {a.get('action','')}", f"   → {a.get('reason','')}"]

        lines += ["", "─" * 60, "EXECUTIVE SUMMARY TABLE", "─" * 60]
        for row in insights.get("exec_table", []):
            lines += [f"• {row.get('Area','')} | {row.get('Finding','')} | Impact: {row.get('Business Impact','')} | Confidence: {row.get('Confidence','')}"]

        lines += ["", "=" * 60, "End of report — generated by MechVision AI", "=" * 60]
        return "\n".join(lines)

    @staticmethod
    def _demo_insights(s: dict) -> dict:
        r2 = s["metrics"].get("R2", 0.72)
        pct = round(r2 * 100, 1)
        return {
            "insights": [
                {
                    "type": "risk",
                    "title": "PTO drive unit is your highest-risk component",
                    "body": (
                        f"The PTO drive unit (ref 56) has the highest complexity score of 4.8 out of 5 "
                        f"and requires 5 connections to other parts. "
                        "A failure here would disconnect power from the entire system — this is your single biggest maintenance liability."
                    ),
                },
                {
                    "type": "risk",
                    "title": "Control box connects to 6 systems — a central failure point",
                    "body": (
                        "The control box (ref 90) touches more parts of the system than any other component. "
                        "If it fails, all 6 connected subsystems lose control simultaneously. "
                        "This warrants redundancy or a backup control pathway."
                    ),
                },
                {
                    "type": "opportunity",
                    "title": "Safety subsystem is lean — a cost-saving opportunity",
                    "body": (
                        f"Your 4 safety components have an average complexity of just 1.8/5, "
                        "meaning they are relatively simple and low-cost to maintain. "
                        "Standardising these parts across your fleet could reduce spare-parts inventory costs by an estimated 15–20%."
                    ),
                },
                {
                    "type": "observation",
                    "title": f"AI model explains {pct}% of complexity variation",
                    "body": (
                        f"The machine learning model predicts component complexity with {pct}% accuracy (R² = {r2}). "
                        "The number of connections is the strongest predictor — more connected parts are more complex. "
                        "This pattern can be used to flag high-risk parts in future designs before manufacturing begins."
                    ),
                },
            ],
            "exec_table": [
                {"Area": "Model accuracy", "Finding": f"AI predicts complexity with {pct}% reliability", "Business Impact": "Reliable for screening", "Confidence": "High" if r2 > 0.65 else "Medium"},
                {"Area": "Top risk component", "Finding": "PTO drive unit — complexity 4.8/5, 5 connections", "Business Impact": "High replacement cost if failure", "Confidence": "High"},
                {"Area": "Cost driver", "Finding": "Fluid subsystem has the largest physical footprint", "Business Impact": "~40% of material cost", "Confidence": "Medium"},
                {"Area": "Safety finding", "Finding": "2 emergency stops present — meets minimum safety standards", "Business Impact": "Regulatory compliant", "Confidence": "High"},
                {"Area": "Recommendation", "Finding": "Prioritise inspection of actuated high-complexity parts", "Business Impact": "Reduce unplanned downtime ~25%", "Confidence": "Medium"},
            ],
            "actions": [
                {"action": "Schedule monthly inspection of PTO unit and control box", "reason": "These two parts have the highest complexity and most connections — a failure here stops the whole system.", "priority": "High"},
                {"action": "Create a standardised spare-parts kit for safety components", "reason": "All 4 safety parts are simple and interchangeable — bulk procurement reduces cost.", "priority": "High"},
                {"action": "Add a backup control pathway for the control box", "reason": "With 6 subsystem connections, the control box is a single point of failure with no redundancy.", "priority": "Medium"},
                {"action": "Use this AI model on new design drawings before approval", "reason": f"At {pct}% accuracy, the model can flag overly complex parts in future designs before costly manufacturing.", "priority": "Low"},
            ],
        }
