from datetime import timedelta
import os
import yaml

# ../config
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "config"
)


def load_default_args():
    with open(os.path.join(CONFIG_DIR, "default_args.yml"), "r") as f:
        default_args = yaml.safe_load(f)
    default_args["retry_delay"] = timedelta(minutes=default_args.get("retry_delay", 5))
    return default_args
