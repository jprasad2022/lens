#!/usr/bin/env python3
"""
Test script to verify Redpanda is working correctly
"""

import asyncio
import os
import sys
sys.path.append('./backend')

from services.kafka_service_impl import KafkaServiceImpl


async def test_redpanda():
    """Test Redpanda functionality"""
    print("🔴 Testing Redpanda Connection...")
    print("=" * 50)
    
    # Override settings for local Redpanda
    os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'localhost:19092'
    os.environ['KAFKA_SECURITY_PROTOCOL'] = 'PLAINTEXT'
    os.environ['USE_REDPANDA'] = 'true'
    os.environ['TEAM_PREFIX'] = 'team1'
    
    try:
        # Initialize Kafka service
        kafka_service = KafkaServiceImpl()
        await kafka_service.initialize()
        print("✅ Connected to Redpanda!")
        
        # Test health check
        print("\n📊 Health Check:")
        health = await kafka_service.health_check()
        print(f"   Healthy: {health.get('healthy', False)}")
        print(f"   Topics: {health.get('topics', [])}")
        
        # Test producing a recommendation request
        print("\n📤 Testing Event Production:")
        
        # 1. Recommendation request
        await kafka_service.produce_reco_request(
            user_id=123,
            request_id="test-001",
            model="als",
            k=10
        )
        print("   ✅ Produced recommendation request")
        
        # 2. Recommendation response
        await kafka_service.produce_reco_response(
            user_id=123,
            request_id="test-001",
            status=200,
            latency_ms=45.2,
            movie_ids=[1, 2, 3, 4, 5],
            model_version="v1.0",
            cached=False
        )
        print("   ✅ Produced recommendation response")
        
        # 3. User interaction event
        await kafka_service.produce_interaction_event({
            "event_type": "movie_watched",
            "user_id": 123,
            "movie_id": 1,
            "recommendation_request_id": "test-001",
            "watch_duration_minutes": 120
        })
        print("   ✅ Produced interaction event")
        
        print("\n✅ All tests passed! Redpanda is working correctly.")
        
        await kafka_service.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Redpanda is running: ./setup-redpanda.sh")
        print("2. Check logs: docker compose -f docker-compose-redpanda.yml logs redpanda")
        print("3. Verify topics exist: docker exec -it redpanda rpk topic list")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_redpanda())
    sys.exit(0 if success else 1)