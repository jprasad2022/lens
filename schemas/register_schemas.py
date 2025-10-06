#!/usr/bin/env python3
"""
Script to register Avro schemas with Confluent Schema Registry
"""

import json
import glob
import os
import requests
from typing import Dict, Any


class SchemaRegistrar:
    def __init__(self, registry_url: str = "http://localhost:8081"):
        self.registry_url = registry_url
        
    def load_schema(self, schema_file: str) -> Dict[str, Any]:
        """Load an Avro schema from file"""
        with open(schema_file, 'r') as f:
            return json.load(f)
    
    def register_schema(self, subject: str, schema: Dict[str, Any]) -> int:
        """Register a schema with the Schema Registry"""
        url = f"{self.registry_url}/subjects/{subject}/versions"
        headers = {"Content-Type": "application/vnd.schemaregistry.v1+json"}
        
        # Wrap schema in the required format
        data = {
            "schema": json.dumps(schema),
            "schemaType": "AVRO"
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            schema_id = response.json()["id"]
            print(f"✓ Registered schema for subject '{subject}' with ID: {schema_id}")
            return schema_id
        else:
            print(f"✗ Failed to register schema for subject '{subject}'")
            print(f"  Status: {response.status_code}")
            print(f"  Error: {response.text}")
            return -1
    
    def set_compatibility(self, subject: str, compatibility: str = "BACKWARD"):
        """Set compatibility level for a subject"""
        url = f"{self.registry_url}/config/{subject}"
        headers = {"Content-Type": "application/vnd.schemaregistry.v1+json"}
        data = {"compatibility": compatibility}
        
        response = requests.put(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print(f"✓ Set compatibility to '{compatibility}' for subject '{subject}'")
        else:
            print(f"✗ Failed to set compatibility for subject '{subject}'")
            print(f"  Status: {response.status_code}")
            print(f"  Error: {response.text}")


def main():
    # Schema Registry configuration
    registry_url = os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    
    # Subject mapping (Kafka topic name to schema file)
    subject_mapping = {
        "user-interactions-value": "user_interaction.avsc",
        "recommendation-requests-value": "recommendation_request.avsc",
        "recommendation-responses-value": "recommendation_response.avsc",
        "model-metrics-value": "model_metrics.avsc",
        "ab-test-events-value": "ab_test_event.avsc"
    }
    
    registrar = SchemaRegistrar(registry_url)
    
    # Check if Schema Registry is available
    try:
        response = requests.get(f"{registry_url}/subjects")
        if response.status_code != 200:
            print(f"Schema Registry not available at {registry_url}")
            return
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to Schema Registry at {registry_url}")
        print("Make sure Schema Registry is running (docker-compose up schema-registry)")
        return
    
    print(f"Connecting to Schema Registry at {registry_url}")
    print("-" * 50)
    
    # Register each schema
    for subject, schema_file in subject_mapping.items():
        schema_path = os.path.join(os.path.dirname(__file__), schema_file)
        
        if not os.path.exists(schema_path):
            print(f"✗ Schema file not found: {schema_path}")
            continue
            
        try:
            schema = registrar.load_schema(schema_path)
            registrar.register_schema(subject, schema)
            registrar.set_compatibility(subject, "BACKWARD")
        except Exception as e:
            print(f"✗ Error processing {schema_file}: {e}")
    
    print("-" * 50)
    print("Schema registration complete!")


if __name__ == "__main__":
    main()