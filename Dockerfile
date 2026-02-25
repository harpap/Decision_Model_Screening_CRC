FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
RUN python -m pip install --index-url https://support.bayesfusion.com/pysmile-A/ pysmile

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "dMS_API:app", "--host", "0.0.0.0", "--port", "8000"]
