from typing import Callable, TypeVar

import anyio
from anyio import Semaphore
from typing_extensions import ParamSpec

from huggingface_inference_toolkit.logging import logger

# To not have too many threads running (which could happen on too many concurrent
# requests, we limit it with a semaphore.
MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)
T = TypeVar("T")
P = ParamSpec("P")


# moves blocking call to asyncio threadpool limited to 1 to not overload the system
# REF: https://stackoverflow.com/a/70929141
async def async_call(handler: Callable[P, T], *args, **kwargs) -> T:
    logger.info("Setting blocking call to async handler")
    async with MAX_THREADS_GUARD:
        logger.info("Async call semaphore passed")
        return await anyio.to_thread.run_sync(handler, *args, **kwargs)
