#!/usr/bin/env python3
"""
Test all imports to identify what might be failing
"""
import sys

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
    except Exception as e:
        print(f"⚠️  {module_name}: {type(e).__name__}: {e}")

print("Testing basic imports...")
test_import("os")
test_import("sys")
test_import("asyncio")
test_import("time")

print("\nTesting FastAPI dependencies...")
test_import("fastapi")
test_import("uvicorn")
test_import("pydantic")
test_import("pydantic_settings")

print("\nTesting other dependencies...")
test_import("prometheus_client")
test_import("redis")
test_import("confluent_kafka")
test_import("pandas")
test_import("numpy")
test_import("sklearn")
test_import("scipy")
test_import("implicit")

print("\nTesting app modules...")
test_import("config")
test_import("config.settings")
test_import("app")
test_import("app.main")
test_import("services")
test_import("routers")

print("\nChecking environment...")
import os
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")  # First 3 entries
print(f"PORT: {os.environ.get('PORT', 'Not set')}")

# Try to get settings
try:
    from config.settings import get_settings
    settings = get_settings()
    print(f"✅ Settings loaded: {settings.app_name}")
except Exception as e:
    print(f"❌ Failed to load settings: {e}")