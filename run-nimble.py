import ollama

response = ollama.systemone(
  model='nimble',
  state='Our checkout has returned 500 errors since 9am.',
  questions={
    'team': {
      'type': 'choice',
      'instructions': 'Which team should handle this ticket?',
      'criteria': {'billing': 'Payments and refunds', 'technical': 'Software errors'},
    },
  },
)
print(response.answers['team'])
