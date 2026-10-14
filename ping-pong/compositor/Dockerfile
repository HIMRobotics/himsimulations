FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir uv==0.12.5
WORKDIR /workspace/project
COPY compositor/pyproject.toml compositor/uv.lock /opt/compositor/
RUN uv sync --project /opt/compositor --frozen --no-dev --no-install-project
ENV PATH="/opt/compositor/.venv/bin:$PATH"
# The encoder writes its in-flight frames and MP4s here; it never needs root.
ENV COMPOSITOR_WORK_DIR=/tmp
COPY compositor/ compositor/
USER 65532:65532
CMD ["python", "-m", "compositor.worker"]
