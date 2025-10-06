# Kafka Schema Registry Configuration

This directory contains Avro schemas for all Kafka topics used in the LENS recommendation system.

## Schemas

### 1. User Interaction Events (`user_interaction.avsc`)
- **Topic**: `user-interactions`
- **Purpose**: Captures all user interactions with the system (views, clicks, ratings, purchases)
- **Key Fields**:
  - `userId`: User identifier
  - `itemId`: Item being interacted with
  - `interactionType`: Type of interaction (view, click, rating, purchase, search, bookmark)
  - `timestamp`: When the interaction occurred

### 2. Recommendation Request (`recommendation_request.avsc`)
- **Topic**: `recommendation-requests`
- **Purpose**: Tracks recommendation requests made to the system
- **Key Fields**:
  - `requestId`: Unique request identifier
  - `userId`: User requesting recommendations
  - `modelName`: Which model to use
  - `numRecommendations`: How many items to recommend

### 3. Recommendation Response (`recommendation_response.avsc`)
- **Topic**: `recommendation-responses`
- **Purpose**: Records recommendation responses from the system
- **Key Fields**:
  - `requestId`: Links to original request
  - `recommendations`: Array of recommended items with scores
  - `processingTime`: How long it took to generate recommendations
  - `status`: Success/partial/error status

### 4. Model Metrics (`model_metrics.avsc`)
- **Topic**: `model-metrics`
- **Purpose**: Tracks model performance metrics
- **Key Fields**:
  - `modelName` & `modelVersion`: Model identification
  - `metricType`: Type of metric (accuracy, precision, recall, latency, etc.)
  - `value`: Metric value
  - `windowStart/End`: Time window for the metric

### 5. A/B Test Events (`ab_test_event.avsc`)
- **Topic**: `ab-test-events`
- **Purpose**: Tracks A/B test assignments and outcomes
- **Key Fields**:
  - `experimentId`: Which experiment
  - `userId`: User in the experiment
  - `variant`: Which variant (control/treatment)
  - `eventType`: Assignment, conversion, engagement, or metric

## Schema Evolution

All schemas are configured with **BACKWARD** compatibility, which means:
- New optional fields can be added
- Defaults must be provided for new fields
- Fields cannot be removed
- Field types cannot be changed

## Usage

### Registering Schemas

1. Start the Schema Registry:
```bash
docker-compose up schema-registry
```

2. Register all schemas:
```bash
cd schemas
python register_schemas.py
```

Or manually register a single schema:
```bash
curl -X POST http://localhost:8081/subjects/user-interactions-value/versions \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  -d '{"schema": "'$(cat user_interaction.avsc | jq -c . | sed 's/"/\\"/g')'"}'
```

### Viewing Registered Schemas

1. List all subjects:
```bash
curl http://localhost:8081/subjects
```

2. Get latest version of a schema:
```bash
curl http://localhost:8081/subjects/user-interactions-value/versions/latest
```

3. Use Kafka UI at http://localhost:8080 to browse schemas visually

## Producer/Consumer Example

### Python Producer with Schema Registry:
```python
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer

value_schema = avro.load('user_interaction.avsc')

avro_producer = AvroProducer({
    'bootstrap.servers': 'localhost:9092',
    'schema.registry.url': 'http://localhost:8081'
}, default_value_schema=value_schema)

# Produce message
avro_producer.produce(
    topic='user-interactions',
    value={
        'eventId': 'evt_123',
        'timestamp': int(time.time() * 1000),
        'userId': 'user_456',
        'sessionId': 'session_789',
        'interactionType': 'click',
        'itemId': 'item_012',
        'itemCategory': 'electronics'
    }
)
```

### Python Consumer with Schema Registry:
```python
from confluent_kafka import avro
from confluent_kafka.avro import AvroConsumer

consumer = AvroConsumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'recommendation-service',
    'schema.registry.url': 'http://localhost:8081',
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(['user-interactions'])

while True:
    msg = consumer.poll(1.0)
    if msg is not None:
        print(f"User {msg.value()['userId']} performed {msg.value()['interactionType']} on item {msg.value()['itemId']}")
```

## Event Replay Strategy

The Schema Registry enables reliable event replay by:

1. **Schema Versioning**: Each schema change gets a new version number
2. **Compatibility Checking**: Ensures consumers can read old messages
3. **Schema ID in Messages**: Each Kafka message includes the schema ID used to serialize it
4. **Historical Schema Access**: Old schema versions remain accessible for replaying historical data

To replay events:
1. Create a new consumer group
2. Set `auto.offset.reset=earliest`
3. The Schema Registry will automatically provide the correct schema version for each message
4. Process messages with their original schema, converting if needed