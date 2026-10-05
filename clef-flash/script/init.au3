curl http://localhost:11434/v1/systemone -d '{
  "model": "clef-flash",
  "state": "The agent wants to submit the attached form.",
  "images": ["<base64-encoded image>"],
  "questions": {
    "complete": {"type": "noul", "instructions": "Are all required fields filled in?"},
    "type": {
      "type": "choice",
      "instructions": "What kind of document is this?",
      "criteria": {"invoice": null, "receipt": null, "contract": null, "other": null}
    }
  }
}'
