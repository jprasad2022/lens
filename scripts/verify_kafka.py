#!/usr/bin/env python3
"""
Verify Kafka topics for Milestone 2
"""
import os
import sys
import json
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Consumer, Producer
import time

def get_kafka_config():
    """Get Kafka configuration"""
    # Try environment variables first
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    # Check if using cloud Kafka (Confluent)
    if os.getenv("KAFKA_API_KEY"):
        return {
            'bootstrap.servers': bootstrap_servers,
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': os.getenv("KAFKA_API_KEY"),
            'sasl.password': os.getenv("KAFKA_API_SECRET"),
        }
    else:
        # Local Kafka
        return {
            'bootstrap.servers': bootstrap_servers,
        }

def create_topics(admin, team_prefix="team1"):
    """Create Kafka topics"""
    topics = [
        f"{team_prefix}.watch",
        f"{team_prefix}.rate", 
        f"{team_prefix}.reco_requests",
        f"{team_prefix}.reco_responses"
    ]
    
    new_topics = [
        NewTopic(topic, num_partitions=3, replication_factor=1)  # rf=1 for local
        for topic in topics
    ]
    
    # Create topics
    fs = admin.create_topics(new_topics)
    
    # Wait for operation to finish
    for topic, f in fs.items():
        try:
            f.result()  # The result itself is None
            print(f"✓ Topic {topic} created")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ Topic {topic} already exists")
            else:
                print(f"✗ Failed to create topic {topic}: {e}")

def list_topics(admin, team_prefix="team1"):
    """List and verify topics"""
    metadata = admin.list_topics(timeout=10)
    
    print("\nKafka Topics:")
    team_topics = []
    for topic in metadata.topics:
        if topic.startswith(team_prefix):
            partitions = metadata.topics[topic].partitions
            print(f"  - {topic} ({len(partitions)} partitions)")
            team_topics.append(topic)
    
    return team_topics

def test_producer_consumer(config, topic):
    """Test producing and consuming messages"""
    # Producer
    producer = Producer(config)
    
    # Test message
    test_message = {
        "user_id": 123,
        "item_id": 456,
        "timestamp": int(time.time()),
        "test": True
    }
    
    # Produce message
    producer.produce(
        topic,
        key=str(test_message["user_id"]),
        value=json.dumps(test_message)
    )
    producer.flush()
    print(f"\n✓ Produced test message to {topic}")
    
    # Consumer
    consumer_config = config.copy()
    consumer_config['group.id'] = 'test-consumer'
    consumer_config['auto.offset.reset'] = 'earliest'
    
    consumer = Consumer(consumer_config)
    consumer.subscribe([topic])
    
    # Poll for message
    print(f"  Consuming from {topic}...")
    msg = consumer.poll(timeout=5.0)
    
    if msg and not msg.error():
        value = json.loads(msg.value())
        print(f"  ✓ Consumed message: {value}")
        consumer.close()
        return True
    else:
        print(f"  ✗ No message received")
        consumer.close()
        return False

def generate_kcat_commands(bootstrap_servers, team_prefix="team1"):
    """Generate kcat commands for verification"""
    print("\nkcat verification commands:")
    print(f"""
# List all topics
kcat -L -b {bootstrap_servers} | grep {team_prefix}

# Consume from a topic (last 10 messages)
kcat -C -b {bootstrap_servers} -t {team_prefix}.watch -o -10 -e

# Produce a test message
echo '{{"user_id": 1, "item_id": 100, "timestamp": '$(date +%s)'}}' | \\
  kcat -P -b {bootstrap_servers} -t {team_prefix}.watch

# Check topic metadata
kcat -L -b {bootstrap_servers} -t {team_prefix}.watch
""")

def generate_consumer_config(config, team_prefix="team1"):
    """Generate consumer configuration snippet"""
    print("\nConsumer Configuration:")
    print(f"""
```python
kafka_config = {{
    "bootstrap.servers": "{config.get('bootstrap.servers', 'localhost:9092')}",
    "group.id": "{team_prefix}-ingestor",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": True,
    "auto.commit.interval.ms": 5000,
    "session.timeout.ms": 6000,
    "max.poll.interval.ms": 300000
}}

# For cloud Kafka (if using Confluent Cloud)
kafka_config_cloud = {{
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "PLAIN",
    "sasl.username": os.getenv("KAFKA_API_KEY"),
    "sasl.password": os.getenv("KAFKA_API_SECRET"),
    "group.id": "{team_prefix}-ingestor",
    "auto.offset.reset": "earliest"
}}
```
""")

def main():
    print("=== Kafka Topic Verification for Milestone 2 ===\n")
    
    # Get configuration
    config = get_kafka_config()
    bootstrap_servers = config['bootstrap.servers']
    team_prefix = os.getenv("TEAM_PREFIX", "team1")
    
    print(f"Team Prefix: {team_prefix}")
    print(f"Bootstrap Servers: {bootstrap_servers}")
    print(f"Security Protocol: {config.get('security.protocol', 'PLAINTEXT')}")
    
    # Create admin client
    try:
        admin = AdminClient(config)
    except Exception as e:
        print(f"\n✗ Failed to connect to Kafka: {e}")
        print("\nMake sure Kafka is running:")
        print("  docker-compose up -d kafka")
        return
    
    # Create topics
    print("\n1. Creating topics...")
    create_topics(admin, team_prefix)
    
    # List topics
    print("\n2. Verifying topics...")
    topics = list_topics(admin, team_prefix)
    
    if len(topics) == 4:
        print(f"\n✓ All 4 topics found for {team_prefix}")
    else:
        print(f"\n✗ Expected 4 topics, found {len(topics)}")
    
    # Test producer/consumer
    if topics:
        print("\n3. Testing producer/consumer...")
        test_topic = f"{team_prefix}.watch"
        if test_producer_consumer(config, test_topic):
            print(f"\n✓ Producer/Consumer test passed")
        else:
            print(f"\n✗ Producer/Consumer test failed")
    
    # Generate commands
    generate_kcat_commands(bootstrap_servers, team_prefix)
    
    # Generate config
    generate_consumer_config(config, team_prefix)
    
    print("\n=== Verification Complete ===")
    
    # Generate report snippet
    print("\nReport snippet for Milestone 2:")
    print(f"""
### Topic Verification
Topics created and verified:
{chr(10).join([f"- {t}" for t in topics])}

Bootstrap Servers: {bootstrap_servers}
Team Prefix: {team_prefix}
""")

if __name__ == "__main__":
    main()