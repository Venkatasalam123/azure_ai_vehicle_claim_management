from azure.monitor.opentelemetry import configure_azure_monitor
import os
from dotenv import load_dotenv
from pathlib import Path


def init_observability():
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)

    configure_azure_monitor(
        connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    )