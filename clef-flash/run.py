pip install typesafe-sdk
export TYPESAFE_BASE_URL=http://localhost:11434
export TYPESAFE_API_KEY=ollama
export TYPESAFE_DEFAULT_MODEL=clef-flash
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

questions = {
    "urgent": Noul(instructions="Is this support request urgent?"),
    "team": Choice(
        instructions="Which team should handle this request?",
        criteria={"billing": "Payments, invoices, and refunds", "technical": "Outages, errors, and configuration", "sales": "Plans and upgrades"},
    ),
    "severity": Score(instructions="How severe is the customer impact?", criteria=["No impact", "Minor", "Major", "Critical"]),
}

with TypeSafeClient(timeout=120) as client:
    result = client.system_one(state="Checkout has been failing for every customer for the last hour.", questions=questions)

print(result.nouls["urgent"].noul)     # 0.998
print(result.choices["team"].choice)   # technical
print(result.scores["severity"].score) # 2.85
