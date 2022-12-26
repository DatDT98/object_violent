import os
import json

with open("config/config.json") as f:
    config_file = json.load(f)


def get_config_variable(varialbe_name):
    value = os.getenv(varialbe_name.upper())
    if value is None:
        value = config_file[varialbe_name.lower()]
    if isinstance(value, str) and value.lower() == "true":
        value = True
    elif isinstance(value, str) and value.lower() == "false":
        value = False
    return value
