# Пересборка TLS Classify в Portainer (код из Git)

После `git pull` / обновления стека из репозитория нужно **пересобрать образы**,
иначе контейнеры продолжат работать со старым кодом.

## Что пересобирать

| Стек / сервис | Когда обязательно |
|---------------|-------------------|
| **manager** (`tls-classify-manager`) | правки очереди, UI, API, reclaim, логи |
| **worker** на **каждой** GPU-машине (`tls-classify-worker`) | правки pipeline, GPU fallback, запись LAS, cancel-watch |

Сейчас после стабилизационных правок нужно пересобрать **и manager, и все worker**.

База SQLite (`manager_data` volume) **сохраняется** при recreate — задачи не пропадут.
`.env` в Portainer/на хосте **не затирается** git pull (он вне репо или в gitignore).

Новые переменные manager (можно добавить в env стека, есть значения по умолчанию):

- `WORKER_LEASE_MINUTES=30` — через сколько минут «молчания» воркера файл вернётся в очередь
- `LOG_TEXT_MAX_CHARS=524288` — максимум символов лога подзадачи в БД

---

## Вариант A — Portainer UI (Git stack)

1. Portainer → **Stacks** → стек management (manager).
2. **Editor** / **Pull and redeploy** (если стек из Git):
   - включите **Re-pull image** / **Rebuild** если есть;
   - либо **Update the stack** с опцией **Re-build image**.
3. Убедитесь, что build context указывает на каталог `cluster/` (как в `deploy/manager/docker-compose.yml`: `context: ../..` относительно `deploy/manager`).
4. Дождитесь успешного deploy, проверьте: `http://<manager>:5357/api/health`.
5. Повторите для **каждого** стека worker на GPU-хостах (тот же git, `deploy/worker`).

Если в Portainer стек только `docker-compose` без build (готовый image из registry) —
сначала соберите и запушьте образы (вариант B), затем в стеке смените tag / Pull.

---

## Вариант B — сборка на хосте из git, потом Portainer

На машине, где есть GPU/CPU и доступ к git-клону:

```bash
cd /path/to/repo/cluster

# Manager
docker build -f manager/Dockerfile -t tls-classify-manager:0.1.0 .
# при необходимости: docker push <your-registry>/tls-classify-manager:0.1.0

# Worker (на машине с NVIDIA Container Toolkit)
docker build -f worker/Dockerfile -t tls-classify-worker:0.1.0 .
# docker push <your-registry>/tls-classify-worker:0.1.0
```

В Portainer → Stack → обновить image tag / **Pull and redeploy**.

---

## Вариант C — docker compose на хосте (без UI)

```bash
# Management
cd /path/to/repo/cluster/deploy/manager
git -C ../.. pull   # или pull всего репо выше
docker compose up -d --build

# На каждой GPU-машине
cd /path/to/repo/cluster/deploy/worker
git -C ../.. pull
docker compose up -d --build
```

`--build` обязателен: иначе поднимется старый локальный image `tls-classify-*:0.1.0`.

---

## Быстрая проверка после обновления

```bash
curl -s http://<manager-ip>:5357/api/health
docker logs tls-classify-manager --tail 50
docker logs tls-classify-worker --tail 50
```

В логе worker при отмене задачи во время расчёта должно появиться
`cancel requested (watch thread)` — признак нового cancel-watch.
