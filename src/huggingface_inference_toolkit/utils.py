import importlib.util
import ipaddress
import os
import sys
from pathlib import Path

import psutil
from starlette.requests import Request

from huggingface_inference_toolkit.const import HF_DEFAULT_PIPELINE_NAME, HF_MODULE_NAME
from huggingface_inference_toolkit.logging import logger

_optimum_available = importlib.util.find_spec("optimum") is not None


def is_optimum_available():
    return False
    # TODO: change when supported
    # return _optimum_available


framework2weight = {
    "pytorch": "pytorch*",
    "tensorflow": "tf*",
    "tf": "tf*",
    "pt": "pytorch*",
    "flax": "flax*",
    "rust": "rust*",
    "onnx": "*onnx*",
    "safetensors": "*safetensors",
    "coreml": "*mlmodel",
    "tflite": "*tflite",
    "savedmodel": "*tar.gz",
    "openvino": "*openvino*",
    "ckpt": "*ckpt",
}


def create_artifact_filter(framework):
    """
    Returns a list of regex pattern based on the DL Framework. which will be used to ignore files when downloading
    """
    ignore_regex_list = list(set(framework2weight.values()))

    pattern = framework2weight.get(framework, None)
    if pattern in ignore_regex_list:
        ignore_regex_list.remove(pattern)
        return ignore_regex_list
    else:
        return []


def check_and_register_custom_pipeline_from_directory(model_dir):
    """
    Checks if a custom pipeline is available and registers it if so.
    """
    # path to custom handler
    custom_module = Path(model_dir).joinpath(HF_DEFAULT_PIPELINE_NAME)
    legacy_module = Path(model_dir).joinpath("pipeline.py")
    custom_pipeline = None
    if custom_module.is_file():
        logger.info(f"Found custom pipeline at {custom_module}")
        spec = importlib.util.spec_from_file_location(HF_MODULE_NAME, custom_module)
        if spec:
            # add the whole directory to path for submodules
            sys.path.insert(0, model_dir)
            # import custom handler
            handler = importlib.util.module_from_spec(spec)
            sys.modules[HF_MODULE_NAME] = handler
            spec.loader.exec_module(handler)
            # init custom handler with model_dir
            custom_pipeline = handler.EndpointHandler(model_dir)
        else:
            logger.info("No spec from file location found for module %s, file %s", HF_MODULE_NAME, custom_module)
    elif legacy_module.is_file():
        logger.warning(
            """You are using a legacy custom pipeline.
            Please update to the new format.
            See documentation for more information."""
        )
        spec = importlib.util.spec_from_file_location("pipeline.PreTrainedPipeline", legacy_module)
        if spec:
            # add the whole directory to path for submodules
            sys.path.insert(0, model_dir)
            # import custom handler
            pipeline = importlib.util.module_from_spec(spec)
            sys.modules["pipeline.PreTrainedPipeline"] = pipeline
            spec.loader.exec_module(pipeline)
            # init custom handler with model_dir
            custom_pipeline = pipeline.PreTrainedPipeline(model_dir)
    else:
        logger.info(f"No custom pipeline found at {custom_module}")

    return custom_pipeline


def convert_params_to_int_or_bool(params):
    """Converts query params to int or bool if possible"""
    for k, v in params.items():
        if v.isnumeric():
            params[k] = int(v)
        if v == "false":
            params[k] = False
        if v == "true":
            params[k] = True
    return params


def should_discard_left() -> bool:
    return os.getenv('DISCARD_LEFT', '0').lower() in ['true', 'yes', '1']


def already_left(request: Request) -> bool:
    """
    Check if the caller has already left without waiting for the answer to come. This can help during burst to relieve
    the pressure on the worker by cancelling jobs whose results don't matter as they won't be fetched anyway
    :param request:
    :return: bool
    """
    # NOTE: Starlette method request.is_disconnected is totally broken, consumes the payload, does not return
    # the correct status. So we use the good old way to identify if the caller is still there.
    # In any case, if we are not sure, we return False
    logger.info("Checking if request caller already left")
    try:
        client = request.client
        host = client.host
        if not host:
            return False

        port = int(client.port)
        host = ipaddress.ip_address(host)

        if port <= 0 or port > 65535:
            logger.warning("Unexpected source port format for caller %s", port)
            return False
        counter = 0
        for connection in psutil.net_connections(kind="tcp"):
            counter += 1
            if connection.status != "ESTABLISHED":
                continue
            if not connection.raddr:
                continue
            if int(connection.raddr.port) != port:
                continue
            if (
                    not connection.raddr.ip
                    or ipaddress.ip_address(connection.raddr.ip) != host
            ):
                continue
            logger.info(
                "Found caller connection still established, caller is most likely still there, %s",
                connection,
            )
            return False
    except Exception as e:
        logger.warning(
            "Unexpected error while checking if caller already left, assuming still there"
        )
        logger.exception(e)
        return False

    logger.info(
        "%d connections checked. No connection found matching to the caller, probably left",
        counter,
    )
    return True
