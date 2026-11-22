# AI-assisted: generated with Claude Code, reviewed and run by the author
# Container for the Streamlit dashboard (used for Choreo; Streamlit Cloud does not need it).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY .streamlit/ ./.streamlit/
COPY artifacts/ ./artifacts/

# Choreo requires a numeric, non-root user ID between 10000 and 20000
RUN useradd --uid 10014 --no-create-home --shell /usr/sbin/nologin appuser
USER 10014

# 8080 is one of the ports Choreo can scale to zero
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/_stcore/health')"

CMD ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
