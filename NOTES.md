source ./delphi/bin/activate

## Start FastAPI server

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Test request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "patient": [
          {"code": "J06", "age_at_event": 27},
          {"code": "E66", "age_at_event": 35},
          {"code": "I10", "age_at_event": 42}
        ]
      }'
```
