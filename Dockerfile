# YAAB runtime image — builds the Rust core, then ships a slim serving layer.
# Local: `docker build -t yaab . && docker run -p 8000:8000 yaab`
# Cloud: the same image runs on Cloud Run / Fargate / Lambda (container) / K8s.

# --- build stage: compile the Rust accelerator into a wheel ----------------
FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin
WORKDIR /src
COPY . .
# Build the yaab-core wheel (falls back to pure-Python at runtime if skipped).
RUN maturin build -m yaab-core/Cargo.toml --release --out /wheels

# --- runtime stage ---------------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY . .
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    && pip install --no-cache-dir '.[litellm,otel]' fastapi uvicorn

ENV YAAB_AGENT="examples.serve_app:agent"
EXPOSE 8000

# Override APP at deploy time: -e YAAB_AGENT="mymodule:agent"
CMD ["sh", "-c", "yaab serve \"$YAAB_AGENT\" --host 0.0.0.0 --port 8000"]
