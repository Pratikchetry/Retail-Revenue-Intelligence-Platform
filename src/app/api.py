"""
Phase 3 — AI Retail Intelligence Platform
FastAPI service layer.
"""

import os
import json
import asyncio
import hashlib
import time as _time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
import pandas as pd

from src.utils.db import engine
from src.utils.logger import get_logger
from src.app.schemas import (
    AskRequest, AskResponse,
    ForecastRequest, ForecastResponse,
    IngestRequest, IngestResponse,
    HealthResponse, MetricsResponse,
)

log = get_logger(__name__)

app = FastAPI(
    title="Retail Revenue Intelligence API",
    description="Multi-agent analytics copilot for UK retail revenue data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Root route for Azure Health Ping (Accepts GET and HEAD)
# ------------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    """Azure App Service pings this route to check if the container is alive."""
    return {"status": "ok", "message": "Retail Intelligence API is running"}

# ------------------------------------------------------------------
# In-Memory Cache (No background warmer to avoid OOM)
# ------------------------------------------------------------------
_cache: dict = {}
_CACHE_TTL = 600  # 10 minutes

def _cache_get(question: str):
    key = hashlib.md5(question.lower().strip().encode()).hexdigest()
    if key in _cache:
        answer, ts = _cache[key]
        if _time.time() - ts < _CACHE_TTL:
            return answer
    return None

def _cache_set(question: str, answer: str):
    key = hashlib.md5(question.lower().strip().encode()).hexdigest()
    _cache[key] = (answer, _time.time())


# ------------------------------------------------------------------
# POST /ask — the main endpoint
# ------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask the multi-agent system a natural language question."""
    log.info("API /ask: '%s'", req.question[:80])
    
    # 1. Check cache first
    cached = _cache_get(req.question)
    if cached:
        log.info("Cache HIT for /ask")
        return AskResponse(
            question=req.question, route="CACHE", answer=cached, 
            recommendation="", evidence="Served from cache.", critic_score=1.0, 
            critic_passes=True, sql="", execution_status="CACHED", result_data=None
        )

    # 2. Run full pipeline if cache miss
    try:
        from src.langgraph.graph import run_agent
        result = run_agent(req.question)
        answer = result.get("answer", "")
        
        # 3. Store in cache
        if answer:
            _cache_set(req.question, answer)
            
        return AskResponse(
            question=req.question,
            route=result.get("route", ""),
            answer=answer,
            recommendation=result.get("recommendation", ""),
            evidence=result.get("evidence", ""),
            critic_score=result.get("critic_score", 0.0),
            critic_passes=result.get("critic_passes", False),
            sql=result.get("sql"),
            execution_status=result.get("execution_status"),
            result_data=result.get("result_data"),
        )
    except Exception as e:
        log.error("/ask failed: %s", str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e)[:300])


# ------------------------------------------------------------------
# POST /chat — streaming endpoint for Vercel AI SDK
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat")
async def chat(req: ChatRequest):
    """Streaming endpoint for Vercel AI SDK (Next.js frontend)."""
    try:
        question = req.messages[-1].content
        history = [msg.dict() for msg in req.messages[:-1]]  # Capture previous turns
        log.info("API /chat: '%s' (History: %d turns)", question[:80], len(history))
        
        # Check cache first
        cached_answer = _cache_get(question)
        if cached_answer:
            log.info("Cache HIT for /chat")
            async def stream_cached():
                words = cached_answer.split()
                for word in words:
                    yield f"{word} "
                    await asyncio.sleep(0.05)
            return StreamingResponse(stream_cached(), media_type="text/plain")
            
        # Run full pipeline if cache miss
        from src.langgraph.graph import run_agent
        
        loop = asyncio.get_event_loop()
        
        # Start the heavy synchronous LangGraph agent in a background thread, passing history
        future = loop.run_in_executor(None, run_agent, question, history)
        
        async def stream_generator():
            # 1. While the agent is thinking, yield spaces to keep the connection alive 
            #    (bypasses Azure 60s idle timeout)
            while not future.done():
                yield " "  # Keep-alive byte
                await asyncio.sleep(3)  # Wait 3 seconds, then send another
                
            # 2. Agent is done! Get the result
            result = future.result()
            answer = result.get("answer", "I could not find an answer.")
            _cache_set(question, answer) # Save to cache for next time
            
            # 3. Yield the actual answer word-by-word
            words = answer.split()
            for word in words:
                yield f"{word} "
                await asyncio.sleep(0.05)
                
        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        log.error("/chat failed: %s", str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e)[:300])


# ------------------------------------------------------------------
# POST /forecast — direct model access (no router)
# ------------------------------------------------------------------
@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """Generate a model-backed revenue forecast."""
    log.info("API /forecast: horizon=%d", req.horizon)
    try:
        from src.ml.generate_forecast import generate_forecast
        from src.utils.db import engine as eng

        forecasts = generate_forecast(horizon=req.horizon)

        # Persist to warehouse
        from src.ml.generate_forecast import save_forecast
        save_forecast(forecasts)

        total = sum(f["forecast_revenue"] for f in forecasts)

        # Load model accuracy
        mae = None
        if os.path.exists("outputs/reports/forecast_summary_xgboost.json"):
            with open("outputs/reports/forecast_summary_xgboost.json") as f:
                mae = json.load(f).get("mae")

        return ForecastResponse(
            horizon=req.horizon,
            total_revenue=round(total, 2),
            avg_daily_revenue=round(total / len(forecasts), 2),
            days=forecasts,
            model_name="XGBoost_V2",
            model_mae=mae,
        )
    except Exception as e:
        log.error("/forecast failed: %s", str(e)[:300])
        endpoint_detail = "Model not trained. Run: python -m src.ml.train_xgboost_v2"
        raise HTTPException(status_code=500, detail=endpoint_detail if "not found" in str(e).lower() else str(e)[:300])


# ------------------------------------------------------------------
# POST /ingest — re-run ingestion
# ------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Trigger warehouse re-ingestion from staging CSVs."""
    log.info("API /ingest: %s", req.csv_name)
    try:
        from src.ingestion.load_warehouse import run_full_ingestion, ANALYTICS_PIPELINE
        run_full_ingestion()
        rows = 0
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT COUNT(*) FROM fact_sales")).scalar()
        return IngestResponse(
            status="complete",
            rows_loaded=rows,
            analytics_tables_rebuilt=len(ANALYTICS_PIPELINE),
        )
    except Exception as e:
        log.error("/ingest failed: %s", str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e)[:300])


# ------------------------------------------------------------------
# GET & HEAD /health
# ------------------------------------------------------------------
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Service + dependency health check (supports HEAD for UptimeRobot)."""
    db_status = "disconnected"
    model_loaded = os.path.exists("src/ml/models/forecast_xgb_v2.pkl")
    chroma_assets = None

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_status = "connected"
            fact_count = conn.execute(text("SELECT COUNT(*) FROM fact_sales")).scalar()
            db_status = f"connected ({fact_count:,} fact_sales rows)"
    except Exception as e:
        log.warning("Health check DB error: %s", str(e)[:100])

    return HealthResponse(
        status="healthy",
        database=db_status,
        model_loaded=model_loaded,
        chroma_assets=chroma_assets,
    )


# ------------------------------------------------------------------
# GET /metrics — KPI snapshot
# ------------------------------------------------------------------
@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    """Business KPI snapshot from the warehouse."""
    try:
        with engine.connect() as conn:
            exec_row = conn.execute(text(
                "SELECT total_revenue, best_month, best_month_revenue, yoy_growth "
                "FROM revenue_executive_summary LIMIT 1"
            )).fetchone()

            total_orders = conn.execute(text(
                "SELECT COUNT(DISTINCT invoice) FROM fact_sales"
            )).scalar()

            anomalies = conn.execute(text(
                "SELECT COUNT(*) FROM ml_anomaly_scores WHERE is_anomaly = TRUE"
            )).scalar()

        mae = None
        if os.path.exists("outputs/reports/forecast_summary_xgboost.json"):
            with open("outputs/reports/forecast_summary_xgboost.json") as f:
                mae = json.load(f).get("mae")

        return MetricsResponse(
            total_revenue=float(exec_row[0]) if exec_row else 0,
            total_orders=total_orders or 0,
            best_month=exec_row[1] if exec_row else "unknown",
            best_month_revenue=float(exec_row[2]) if exec_row else 0,
            yoy_growth=float(exec_row[3]) if exec_row and exec_row[3] is not None else None,
            forecast_model_mae=mae,
            anomalies_detected=anomalies or 0,
        )
    except Exception as e:
        log.error("/metrics failed: %s", str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e)[:300])


@app.get("/custom/monthly-revenue")
def monthly_revenue():
    """Monthly revenue time series for the dashboard chart."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(
                "SELECT month_date, month_name || ' ' || year AS label, "
                "total_revenue, total_orders "
                "FROM revenue_monthly_summary ORDER BY month_date"
            ), conn)
        return df.to_dict(orient="records")
    except Exception as e:
        log.error("/custom/monthly-revenue failed: %s", str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e)[:300])


@app.get("/custom/segment-revenue")
def segment_revenue():
    """Customer segment revenue breakdown for the dashboard."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(
                "SELECT customer_segment, customer_count, total_revenue, "
                "avg_order_revenue "
                "FROM customer_segment_revenue ORDER BY total_revenue DESC"
            ), conn)
        return df.to_dict(orient="records")
    except Exception as e:
        log.error("/custom/segment-revenue failed: %s", str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e)[:300])