import os, time
def model_version(): return os.getenv("MODEL_VERSION", time.strftime("%Y%m%d_%H%M%S"))
