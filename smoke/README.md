# Проверка без GPU (необязательно)

Из папки `cluster/`:

```bash
python smoke/prepare_smoke_data.py
docker compose -f docker-compose.smoke.yml up -d --build
python smoke/run_smoke.py http://127.0.0.1:5357
```

Создаёт тестовые файлы, гоняет mock-worker (копирование вместо GPU) и проверяет skip/cancel.
