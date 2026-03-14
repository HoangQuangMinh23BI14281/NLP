Run demo API:

1. `cd demo/api`
2. `python run_demo.py`

Endpoints:

- Frontend UI: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Labels metadata: `http://127.0.0.1:8000/labels`
- Job match API: `http://127.0.0.1:8000/match`

Sample test text:

`We are looking for a Senior Java Developer with 5 years of experience in Spring Boot and AWS, located in Berlin, salary up to $5000/month.`

TODO (not implemented in current demo/api):

- PDF upload parsing pipeline.
- Frontend still uses `/predict`; integrate `/match` view if you want company ranking cards.
- Confidence score is currently estimated from token logits; sequence-level CRF confidence is a possible future enhancement.