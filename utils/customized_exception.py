import json

with open("config/error_messages.json") as f:
    error_messages = json.load(f)


class BadRequestException(Exception):
    def __init__(self, code, field, *args):
        self.code = code
        self.field = field
        error_message = error_messages[code]
        if args is not None:
            self.message = error_message.format(*args)


class InternalException(Exception):
    def __init__(self, message):
        self.message = message
