curl http://localhost:11434/v1/systemone -d '{
  "model": "clef-flash",
  "state": "Checkout has been failing for every customer for the last hour.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices, and refunds",
        "technical": "Outages, errors, and configuration",
        "sales": "Plans and upgrades"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the customer impact?",
      "criteria": ["No impact", "Minor", "Major", "Critical"]
    }
  }
}'
{
  "model": "clef-flash",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.998},
    "team": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {"billing": 0.004, "technical": 0.992, "sales": 0.004},
      "confidence": 0.947
    },
    "severity": {
      "type": "score",
      "score": 2.851,
      "legend": {"0": "No impact", "1": "Minor", "2": "Major", "3": "Critical"},
      "probabilities": {"0": 0.002, "1": 0.014, "2": 0.271, "3": 0.713},
      "confidence": 0.631
    }
  },
  "usage": {"input_tokens": 486, "output_tokens": 3}
}
