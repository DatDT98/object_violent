import json

import grpc
from utils.customized_exception import BadRequestException, InternalException
import logging
import traceback

logger = logging.getLogger(__name__)


def handle_error_status(exception, context):
    logger.info(exception)
    if isinstance(exception, BadRequestException):
        return create_bad_request_error_status(exception, context)
    elif isinstance(exception, InternalException):
        traceback.print_exc()
        return create_internal_error_status(exception.message, context)
    else:
        traceback.print_exc()
        return create_internal_error_status(exception, context)


def create_bad_request_error_status(exception: BadRequestException, context):
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(json.dumps({"error_code": exception.code, "message": exception.message}))


def create_internal_error_status(exception, context):
    context.set_code(grpc.StatusCode.INTERNAL)
    context.set_details(str(exception))
