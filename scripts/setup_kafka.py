#!/usr/bin/env python3
"""
Setup and verify Kafka topics for MovieLens
"""

import os
import sys
import argparse
from confluent_kafka import Consumer, Producer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic

def get_kafka_config():
    """Get Kafka configuration from environment."""
    config = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    }
    
    # Add authentication if using Confluent Cloud
    if config["security.protocol"] == "SASL_SSL":
        config.update({
            "sasl.mechanisms": "PLAIN",
            "sasl.username": os.getenv("KAFKA_API_KEY"),
            "sasl.password": os.getenv("KAFKA_API_SECRET"),
        })
    
    return config

def create_topics(admin_client, team_prefix):
    """Create required topics."""
    topics = [
        f"{team_prefix}.watch",
        f"{team_prefix}.rate",
        f"{team_prefix}.reco_requests",
        f"{team_prefix}.reco_responses",
    ]
    
    # Check existing topics
    existing_topics = admin_client.list_topics(timeout=10).topics
    
    # Create missing topics
    new_topics = []
    for topic in topics:
        if topic not in existing_topics:
            new_topics.append(NewTopic(
                topic,
                num_partitions=3,
                replication_factor=3,
                config={
                    "retention.ms": "604800000",  # 7 days
                    "compression.type": "gzip",
                }
            ))
    
    if new_topics:
        fs = admin_client.create_topics(new_topics)
        for topic, f in fs.items():
            try:
                f.result()
                print(f"✅ Created topic: {topic}")
            except Exception as e:
                print(f"❌ Failed to create topic {topic}: {e}")
    else:
        print("✅ All topics already exist")
    
    return topics

def verify_topics(admin_client, topics):
    """Verify topics exist and show metadata."""
    print("\n📋 Topic Verification:")
    print("-" * 50)
    
    metadata = admin_client.list_topics(timeout=10)
    
    for topic in topics:
        if topic in metadata.topics:
            topic_metadata = metadata.topics[topic]
            print(f"\n✅ Topic: {topic}")
            print(f"   Partitions: {len(topic_metadata.partitions)}")
            print(f"   Replicas: {len(list(topic_metadata.partitions.values())[0].replicas)}")
        else:
            print(f"\n❌ Topic not found: {topic}")

def test_producer(config, team_prefix):
    """Test producing a message."""
    print("\n📤 Testing Producer:")
    print("-" * 50)
    
    producer = Producer(config)
    
    test_topic = f"{team_prefix}.reco_requests"
    test_message = '{"test": true, "message": "Kafka setup verification"}'
    
    def delivery_report(err, msg):
        if err is not None:
            print(f"❌ Message delivery failed: {err}")
        else:
            print(f"✅ Message delivered to {msg.topic()} [{msg.partition()}]")
    
    producer.produce(
        test_topic,
        key="test",
        value=test_message,
        callback=delivery_report
    )
    
    producer.flush(timeout=10)

def test_consumer(config, team_prefix):
    """Test consuming messages."""
    print("\n📥 Testing Consumer:")
    print("-" * 50)
    
    consumer_config = config.copy()
    consumer_config.update({
        "group.id": f"{team_prefix}-setup-test",
        "auto.offset.reset": "earliest",
    })
    
    consumer = Consumer(consumer_config)
    
    # Subscribe to all topics
    topics = [
        f"{team_prefix}.watch",
        f"{team_prefix}.rate",
        f"{team_prefix}.reco_requests",
        f"{team_prefix}.reco_responses",
    ]
    
    consumer.subscribe(topics)
    
    print(f"✅ Successfully subscribed to topics: {topics}")
    print("\nSample consumer configuration:")
    print(f"  Bootstrap servers: {config['bootstrap.servers']}")
    print(f"  Security protocol: {config['security.protocol']}")
    print(f"  Group ID: {consumer_config['group.id']}")
    
    consumer.close()

def main():
    parser = argparse.ArgumentParser(description="Setup Kafka topics for MovieLens")
    parser.add_argument("--team-prefix", default=os.getenv("TEAM_PREFIX", "team1"),
                        help="Team prefix for topics")
    parser.add_argument("--create", action="store_true",
                        help="Create topics if they don't exist")
    parser.add_argument("--verify", action="store_true",
                        help="Verify topics exist")
    parser.add_argument("--test", action="store_true",
                        help="Test producer and consumer")
    
    args = parser.parse_args()
    
    # Get Kafka configuration
    config = get_kafka_config()
    
    # Create admin client
    admin_client = AdminClient(config)
    
    print(f"🚀 Kafka Setup for Team: {args.team_prefix}")
    print(f"   Bootstrap servers: {config['bootstrap.servers']}")
    print(f"   Security protocol: {config['security.protocol']}")
    
    # Define topics
    topics = [
        f"{args.team_prefix}.watch",
        f"{args.team_prefix}.rate",
        f"{args.team_prefix}.reco_requests",
        f"{args.team_prefix}.reco_responses",
    ]
    
    # Create topics if requested
    if args.create:
        create_topics(admin_client, args.team_prefix)
    
    # Verify topics
    if args.verify or args.create:
        verify_topics(admin_client, topics)
    
    # Test producer/consumer
    if args.test:
        test_producer(config, args.team_prefix)
        test_consumer(config, args.team_prefix)
    
    # Show kcat commands for verification
    print("\n📝 Verify with kcat:")
    print("-" * 50)
    print(f"# List topics:")
    print(f"kcat -L -b {config['bootstrap.servers']}")
    
    if config["security.protocol"] == "SASL_SSL":
        print(f"\n# Consume from topic (with auth):")
        print(f"kcat -C -b {config['bootstrap.servers']} \\")
        print(f"  -X security.protocol=SASL_SSL \\")
        print(f"  -X sasl.mechanisms=PLAIN \\")
        print(f"  -X sasl.username=$KAFKA_API_KEY \\")
        print(f"  -X sasl.password=$KAFKA_API_SECRET \\")
        print(f"  -t {args.team_prefix}.reco_requests")
    else:
        print(f"\n# Consume from topic:")
        print(f"kcat -C -b {config['bootstrap.servers']} -t {args.team_prefix}.reco_requests")

if __name__ == "__main__":
    main()