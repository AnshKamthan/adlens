# adlens

Mini ad-creative classification service used as the hands-on project in
`production-ai-engineering-crash-course.md`. FastAPI + SQLAlchemy (SQLite/Postgres) +
Redis cache + OpenAI-compatible LLM client + offline evals + CI.

    cp .env.example .env
    uv sync
    make test          # 26 tests, no network, no API key
    make eval-compare  # v1 vs v2 prompt on the golden set
    make dev           # http://localhost:8000/docs
