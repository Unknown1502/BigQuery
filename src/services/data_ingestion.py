"""
Compatibility module for data-ingestion package
This module provides access to the data-ingestion package with hyphenated name
"""

import importlib.util
import sys
from pathlib import Path

# Get the path to the data-ingestion directory
data_ingestion_path = Path(__file__).parent / "data-ingestion"

# Import the connectors module
connectors_spec = importlib.util.spec_from_file_location(
    "connectors", 
    data_ingestion_path / "connectors" / "__init__.py"
)
connectors_module = importlib.util.module_from_spec(connectors_spec)

# Import the collectors module
collectors_spec = importlib.util.spec_from_file_location(
    "collectors", 
    data_ingestion_path / "collectors" / "__init__.py"
)
collectors_module = importlib.util.module_from_spec(collectors_spec)

# Import the street_camera_connector
street_camera_connector_spec = importlib.util.spec_from_file_location(
    "street_camera_connector",
    data_ingestion_path / "connectors" / "street_camera_connector.py"
)
street_camera_connector_module = importlib.util.module_from_spec(street_camera_connector_spec)
street_camera_connector_spec.loader.exec_module(street_camera_connector_module)

# Import the street_image_dataset_collector
street_image_dataset_collector_spec = importlib.util.spec_from_file_location(
    "street_image_dataset_collector",
    data_ingestion_path / "collectors" / "street_image_dataset_collector.py"
)
street_image_dataset_collector_module = importlib.util.module_from_spec(street_image_dataset_collector_spec)
street_image_dataset_collector_spec.loader.exec_module(street_image_dataset_collector_module)

# Create a namespace for easier access
class DataIngestionNamespace:
    class connectors:
        street_camera_connector = street_camera_connector_module
        
    class collectors:
        street_image_dataset_collector = street_image_dataset_collector_module

# Make the modules available
data_ingestion = DataIngestionNamespace()
