#!/bin/bash

# Create Kafka topics for team1
KAFKA_CONTAINER="lens-kafka-1"
BOOTSTRAP_SERVER="kafka:9092"

echo "Creating Kafka topics for team1..."

# Create topics
docker exec $KAFKA_CONTAINER kafka-topics --create --bootstrap-server $BOOTSTRAP_SERVER --topic team1.watch --partitions 3 --replication-factor 1 2>/dev/null
docker exec $KAFKA_CONTAINER kafka-topics --create --bootstrap-server $BOOTSTRAP_SERVER --topic team1.rate --partitions 3 --replication-factor 1 2>/dev/null
docker exec $KAFKA_CONTAINER kafka-topics --create --bootstrap-server $BOOTSTRAP_SERVER --topic team1.reco_requests --partitions 3 --replication-factor 1 2>/dev/null
docker exec $KAFKA_CONTAINER kafka-topics --create --bootstrap-server $BOOTSTRAP_SERVER --topic team1.reco_responses --partitions 3 --replication-factor 1 2>/dev/null

echo -e "\nListing team1 topics:"
docker exec $KAFKA_CONTAINER kafka-topics --list --bootstrap-server $BOOTSTRAP_SERVER | grep team1

echo -e "\nTopic details:"
docker exec $KAFKA_CONTAINER kafka-topics --describe --bootstrap-server $BOOTSTRAP_SERVER --topic team1.watch