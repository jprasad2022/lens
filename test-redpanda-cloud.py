#!/usr/bin/env python3
"""
Test Redpanda Cloud connection
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

from services.kafka_service_impl import KafkaServiceImpl


async def test_redpanda_cloud():
    """Test Redpanda Cloud connection"""
    print("☁️  Testing Redpanda Cloud Connection...")
    print("=" * 50)
    
    # Load cloud configuration
    env_file = backend_path / ".env.cloud"
    if env_file.exists():
        print(f"Loading configuration from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value.strip()
    
    # Mask password in output
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'not set')
    username = os.getenv('KAFKA_SASL_USERNAME', 'not set')
    has_password = 'yes' if os.getenv('KAFKA_SASL_PASSWORD') else 'no'
    
    print(f"\nConfiguration:")
    print(f"  Bootstrap Servers: {bootstrap_servers}")
    print(f"  Username: {username}")
    print(f"  Password Set: {has_password}")
    print(f"  Security Protocol: {os.getenv('KAFKA_SECURITY_PROTOCOL', 'not set')}")
    
    try:
        # Initialize Kafka service
        kafka_service = KafkaServiceImpl()
        await kafka_service.initialize()
        print("\n✅ Connected to Redpanda Cloud!")
        
        # Test health check
        print("\n📊 Health Check:")
        health = await kafka_service.health_check()
        print(f"   Healthy: {health.get('healthy', False)}")
        print(f"   Connected: {health.get('connected', False)}")
        
        # List existing topics
        print(f"\n📋 Topics in cloud:")
        topics = health.get('topics', {})
        if isinstance(topics, dict):
            for topic in topics.values():
                print(f"   - {topic}")
        elif isinstance(topics, list):
            for topic in topics:
                print(f"   - {topic}")
        
        # Test producing an event
        print("\n📤 Testing Event Production:")
        await kafka_service.produce_reco_request(
            user_id=999,
            request_id="cloud-test-001",
            model="als",
            k=5
        )
        print("   ✅ Successfully produced test event to cloud!")
        
        print("\n✅ Redpanda Cloud is working correctly!")
        print("\n🎉 You can now deploy your app with these credentials!")
        
        await kafka_service.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Redpanda Cloud credentials in backend/.env.cloud")
        print("2. Make sure your service account has the right permissions")
        print("3. Verify the bootstrap server URL is correct")
        print("4. Check if your cluster is running in Redpanda Cloud console")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_redpanda_cloud())
    sys.exit(0 if success else 1)