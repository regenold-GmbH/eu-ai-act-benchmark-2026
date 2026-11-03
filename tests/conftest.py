import os

# Avoid LiteLLM's optional pricing-metadata download during offline tests.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
