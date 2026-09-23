from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.transformer_predictor import TransformerPredictor


app = FastAPI(
    title="News Intelligence Sentiment API",
    description="Target-dependent sentiment analysis using DistilBERT",
    version="1.0.0",
)


# Load the model once when the API starts.
predictor = TransformerPredictor()


class PredictionRequest(BaseModel):
    sentence: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)

    target_from: int | None = Field(
        default=None,
        ge=0
    )

    target_to: int | None = Field(
        default=None,
        ge=0
    )


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest] = Field(
        ...,
        min_length=1
    )


@app.get("/")
def root():
    return {
        "message": "News Intelligence Sentiment API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "distilbert-base-uncased"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    # Both offsets must be supplied together.
    if (
        (request.target_from is None)
        != (request.target_to is None)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "target_from and target_to must "
                "either both be provided or both be omitted."
            )
        )

    # target_to must be greater than target_from.
    if (
        request.target_from is not None
        and request.target_to is not None
        and request.target_to <= request.target_from
    ):
        raise HTTPException(
            status_code=400,
            detail="target_to must be greater than target_from."
        )

    # Offsets must be inside the sentence.
    if request.target_to is not None:
        if request.target_to > len(request.sentence):
            raise HTTPException(
                status_code=400,
                detail="target_to exceeds sentence length."
            )

    try:
        result = predictor.predict(
            sentence=request.sentence,
            target=request.target,
            target_from=request.target_from,
            target_to=request.target_to
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    return {
        "sentence": result["sentence"],
        "target": result["target"],
        "marked_sentence": result["marked_sentence"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    }


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):

    results = []

    for item in request.requests:

        if (
            (item.target_from is None)
            != (item.target_to is None)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "target_from and target_to must "
                    "both be provided or both be omitted."
                )
            )

        if (
            item.target_from is not None
            and item.target_to is not None
            and item.target_to <= item.target_from
        ):
            raise HTTPException(
                status_code=400,
                detail="target_to must be greater than target_from."
            )

        if item.target_to is not None:
            if item.target_to > len(item.sentence):
                raise HTTPException(
                    status_code=400,
                    detail="target_to exceeds sentence length."
                )

        try:
            result = predictor.predict(
                sentence=item.sentence,
                target=item.target,
                target_from=item.target_from,
                target_to=item.target_to
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc)
            )

        results.append({
            "sentence": result["sentence"],
            "target": result["target"],
            "marked_sentence": result["marked_sentence"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        })

    return {
        "count": len(results),
        "results": results
    }