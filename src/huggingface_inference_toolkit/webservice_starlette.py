import asyncio
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import orjson
from anyio import Semaphore
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from huggingface_inference_toolkit import idle
from huggingface_inference_toolkit.async_utils import MAX_CONCURRENT_THREADS, MAX_THREADS_GUARD, async_call
from huggingface_inference_toolkit.const import (
    HF_FRAMEWORK,
    HF_HUB_TOKEN,
    HF_MODEL_DIR,
    HF_MODEL_ID,
    HF_REVISION,
    HF_TASK,
)
from huggingface_inference_toolkit.env_utils import api_inference_compat
from huggingface_inference_toolkit.handler import (
    get_inference_handler_either_custom_or_default_handler,
)
from huggingface_inference_toolkit.latency_guard import latency_guard
from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.serialization.json_utils import Jsoner
from huggingface_inference_toolkit.utils import convert_params_to_int_or_bool, should_discard_left
from huggingface_inference_toolkit.vertex_ai_utils import _load_repository_from_gcs

INFERENCE_HANDLERS = {}
INFERENCE_HANDLERS_SEMAPHORE = Semaphore(1)
MODEL_DOWNLOADED = False
MODEL_DL_SEMAPHORE = Semaphore(1)


async def drain_queue():
    """Shutdown hook: wait for all in-flight and queued inference requests to complete."""
    pending = MAX_THREADS_GUARD.statistics().tasks_waiting
    active = MAX_CONCURRENT_THREADS - MAX_THREADS_GUARD.value
    if pending or active:
        logger.info("Shutdown: draining %d active + %d queued request(s)", active, pending)
        while MAX_THREADS_GUARD.value < MAX_CONCURRENT_THREADS or MAX_THREADS_GUARD.statistics().tasks_waiting > 0:
            await asyncio.sleep(0.1)
    logger.info("Shutdown: queue drained")


async def prepare_model_artifacts():
    global INFERENCE_HANDLERS

    if idle.UNLOAD_IDLE:
        asyncio.create_task(idle.live_check_loop(), name="live_check_loop")
    else:
        await async_call(_eager_model_dl)
        logger.info(f"Initializing model from directory:{HF_MODEL_DIR}")
        # 2. determine correct inference handler
        inference_handler = await get_inference_handler_either_custom_or_default_handler(
            HF_MODEL_DIR, task=HF_TASK
        )
        INFERENCE_HANDLERS[HF_TASK] = inference_handler
    logger.info("Model initialized successfully")


def _eager_model_dl():
    logger.info("Model download")
    global MODEL_DOWNLOADED
    from huggingface_inference_toolkit.heavy_utils import load_repository_from_hf
    # 1. check if model artifacts available in HF_MODEL_DIR
    if len(list(Path(HF_MODEL_DIR).glob("**/*"))) <= 0:
        # 2. if not available, try to load from HF_MODEL_ID
        if HF_MODEL_ID is not None:
            load_repository_from_hf(
                repository_id=HF_MODEL_ID,
                target_dir=HF_MODEL_DIR,
                framework=HF_FRAMEWORK,
                revision=HF_REVISION,
                hf_hub_token=HF_HUB_TOKEN,
            )
        # 3. check if in Vertex AI environment and load from GCS
        # If artifactUri not on Model Creation not set returns an empty string
        elif len(os.environ.get("AIP_STORAGE_URI", "")) > 0:
            _load_repository_from_gcs(
                os.environ["AIP_STORAGE_URI"], target_dir=HF_MODEL_DIR
            )
        # 4. if not available, raise error
        else:
            raise ValueError(
                f"""Can't initialize model.
                    Please set env HF_MODEL_DIR or provider a HF_MODEL_ID.
                    Provided values are:
                    HF_MODEL_DIR: {HF_MODEL_DIR} and HF_MODEL_ID:{HF_MODEL_ID}"""
            )
    else:
        logger.info("Model already downloaded in %s", HF_MODEL_DIR)
    logger.info("Model successfully downloaded")
    MODEL_DOWNLOADED = True


@asynccontextmanager
async def lifespan(app):
    # Starlette 1.0 removed `on_startup` / `on_shutdown` in favor of the
    # ASGI lifespan protocol. We run the model-artifact preparation once at
    # startup; there is no per-process shutdown work to do.
    await prepare_model_artifacts()
    yield
    await drain_queue()


async def health(request):
    return PlainTextResponse("Ok")


# Report Prometheus metrics
# inf_batch_current_size: Current number of requests being processed
# inf_queue_size: Number of requests waiting in the queue
async def metrics(request):
    batch_current_size = MAX_CONCURRENT_THREADS - MAX_THREADS_GUARD.value
    queue_size = MAX_THREADS_GUARD.statistics().tasks_waiting
    return PlainTextResponse(
        f"inf_batch_current_size {batch_current_size}\n"
        f"inf_queue_size {queue_size}\n"
        f"inf_accepting {int(latency_guard.accepting)}\n"
        f"inf_auto_frozen {int(latency_guard.auto_frozen)}\n"
    )


async def predict(request):
    if not latency_guard.accepting:
        return Response(
            Jsoner.serialize({"error": "Service temporarily unavailable, overload detected"}),
            status_code=503,
            media_type="application/json",
        )

    start_time = perf_counter()

    async with idle.request_witnesses():
        logger.debug("Received request, scope %s", request.scope)

        global INFERENCE_HANDLERS

        if not MODEL_DOWNLOADED:
            async with MODEL_DL_SEMAPHORE:
                if not MODEL_DOWNLOADED:
                    logger.info("Model dl semaphore acquired")
                    await async_call(_eager_model_dl)
                    logger.info("Model dl semaphore released")
        try:
            task = request.path_params.get("task", HF_TASK)
            # extracts content from request
            content_type = request.headers.get("content-Type", os.environ.get("DEFAULT_CONTENT_TYPE", "")).lower()
            # try to deserialize payload
            deserialized_body = ContentType.get_deserializer(content_type, task).deserialize(
                await request.body()
            )
            # checks if input schema is correct
            if "inputs" not in deserialized_body and "instances" not in deserialized_body:
                raise ValueError(
                    f"Body needs to provide a inputs key, received: {orjson.dumps(deserialized_body)}"
                )

            # Decode base64 audio inputs before running inference
            if "parameters" in deserialized_body and HF_TASK in {
                "automatic-speech-recognition",
                "audio-classification",
            }:
                # Be more strict on base64 decoding, the provided string should valid base64 encoded data
                deserialized_body["inputs"] = base64.b64decode(
                    deserialized_body["inputs"], validate=True
                )

            # check for query parameter and add them to the body
            if request.query_params and "parameters" not in deserialized_body:
                deserialized_body["parameters"] = convert_params_to_int_or_bool(
                    dict(request.query_params)
                )

            # We lazily load pipelines for alt tasks

            if task == "feature-extraction" and HF_TASK in [
                "sentence-similarity",
                "sentence-embeddings",
                "sentence-ranking",
            ]:
                task = "sentence-embeddings"
            inference_handler = INFERENCE_HANDLERS.get(task)
            if not inference_handler:
                async with INFERENCE_HANDLERS_SEMAPHORE:
                    if task not in INFERENCE_HANDLERS:
                        inference_handler = await get_inference_handler_either_custom_or_default_handler(
                            HF_MODEL_DIR, task=task)
                        INFERENCE_HANDLERS[task] = inference_handler
                    else:
                        inference_handler = INFERENCE_HANDLERS[task]

            logger.info("Calling inference handler prediction routine")
            # run async not blocking call, skipping it if the caller is gone by the time a slot frees
            pred = await async_call(
                inference_handler, deserialized_body, request=request if should_discard_left() else None
            )

            # log request time
            end_time = perf_counter()
            logger.info(
                f"POST {request.url.path} Total request duration: {(end_time-start_time) *1000:.2f} ms"
            )

            if pred is None:
                logger.info("No content returned as caller already left")
                return Response(status_code=204)

            # response extracts content from request
            accept = request.headers.get("accept")
            if accept is None or accept == "*/*":
                accept = os.environ.get("DEFAULT_ACCEPT", "application/json")
            logger.info("Request accepts %s", accept)
            # deserialized and resonds with json
            serialized_response_body = ContentType.get_serializer(accept).serialize(
                pred, accept
            )
            return Response(serialized_response_body, media_type=accept)
        except Exception as e:
            logger.exception(e)
            return Response(
                Jsoner.serialize({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )


# Create app based on which cloud environment is used
if os.getenv("AIP_MODE", None) == "PREDICTION":
    logger.info("Running in Vertex AI environment")
    # extract routes from environment variables
    _predict_route = os.getenv("AIP_PREDICT_ROUTE", None)
    _health_route = os.getenv("AIP_HEALTH_ROUTE", None)
    if _predict_route is None or _health_route is None:
        raise ValueError(
            "AIP_PREDICT_ROUTE and AIP_HEALTH_ROUTE need to be set in Vertex AI environment"
        )

    app = Starlette(
        debug=False,
        routes=[
            Route(_health_route, health, methods=["GET"]),
            Route(_predict_route, predict, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
else:
    routes = [
        Route("/", health, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/", predict, methods=["POST"]),
        Route("/predict", predict, methods=["POST"]),
        Route("/metrics", metrics, methods=["GET"]),
    ]
    if api_inference_compat():
        routes.append(
            Route("/pipeline/{task:path}", predict, methods=["POST"])
        )
    app = Starlette(
        debug=False,
        routes=routes,
        lifespan=lifespan,
    )
