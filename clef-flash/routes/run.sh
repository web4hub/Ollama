curl http://localhost:11434/v1/systemone -d '{
  "model": "clef-flash",
  "state": "Design a sharded database schema for a payments ledger.",
  "questions": {
    "model": {
      "type": "choice",
      "instructions": "Which model should answer this prompt?",
      "criteria": {"gemma4": "Small model", "gpt-6": "Large model"}
    }
  }
}'
