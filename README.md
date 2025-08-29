# Trade Project (Model Service + Backend)

## Quick start
```bash
docker compose up --build
# Backend: http://localhost:8000/health
# Model:   http://localhost:9000/health
```

### Smoke tests
```bash
curl :8000/health
curl -X POST :8000/api/ai/consensus -H 'content-type: application/json' \
  -d '{"symbol":"BTCUSDT","features":{"mom_20":1.0,"rv_5":0.2}}'
curl -X POST :8000/api/trade/execute -H 'content-type: application/json' \
  -d '{"symbol":"BTCUSDT","side":"buy","qtyUsd":2000,"confidence":0.9}'
```
