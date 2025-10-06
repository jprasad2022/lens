# Event Replay Strategy

## Overview

The LENS recommendation system implements a comprehensive event replay strategy to enable:
- Historical data reprocessing
- Model retraining with updated algorithms
- Debugging and troubleshooting
- Compliance and audit requirements
- Disaster recovery
- A/B test analysis

## Architecture Components

### 1. Event Storage Layer

#### Kafka Topics
All events are stored in Kafka with the following configuration:
- **Retention Period**: 30 days for real-time topics, 365 days for replay topics
- **Replication Factor**: 3 (production), 1 (development)
- **Partitioning**: By user_id for ordered replay per user

#### Long-term Storage
Events older than 30 days are archived to:
- **Primary**: Google Cloud Storage (GCS) in Parquet format
- **Format**: Daily partitioned Parquet files with Snappy compression
- **Schema**: Managed by Confluent Schema Registry
- **Retention**: 3 years for compliance

### 2. Schema Evolution

#### Schema Registry Integration
- All events use Avro schemas registered in Confluent Schema Registry
- Backward compatibility enforced for all schemas
- Schema versions tracked with each event
- Automatic schema evolution handling during replay

#### Schema Version Mapping
```json
{
  "event_type": "user_interaction",
  "schema_versions": {
    "v1": "2024-01-01 to 2024-03-15",
    "v2": "2024-03-16 to 2024-06-30", 
    "v3": "2024-07-01 to present"
  }
}
```

### 3. Replay Mechanisms

#### Real-time Replay
For events within Kafka retention window (30 days):

```python
# Example: Replay last 7 days of user interactions
replay_consumer = ReplayConsumer(
    topics=['user-interactions'],
    start_timestamp=datetime.now() - timedelta(days=7),
    end_timestamp=datetime.now(),
    consumer_group='replay-ml-training-20240315'
)

for event in replay_consumer:
    process_event(event)
```

#### Historical Replay
For events beyond Kafka retention:

```python
# Example: Replay Q1 2024 data from GCS
historical_replay = HistoricalReplay(
    bucket='lens-events-archive',
    start_date='2024-01-01',
    end_date='2024-03-31',
    event_types=['user_interaction', 'recommendation_response']
)

for batch in historical_replay.get_batches():
    process_batch(batch)
```

#### Selective Replay
Replay specific subsets of events:

```python
# Example: Replay only failed recommendations
selective_replay = SelectiveReplay(
    filter={
        'event_type': 'recommendation_response',
        'status': 'error',
        'date_range': ['2024-03-01', '2024-03-15']
    }
)
```

## Replay Scenarios

### 1. Model Retraining
**Use Case**: Retrain models with historical data after algorithm improvements

**Process**:
1. Create new consumer group: `model-retrain-{timestamp}`
2. Set replay window (typically 90 days)
3. Filter relevant event types
4. Process events through updated training pipeline
5. Validate new model performance
6. Deploy if metrics improve

**Implementation**:
```yaml
replay_config:
  name: "collaborative-filter-v2-training"
  start_offset: "90d"
  events:
    - user_interaction
    - recommendation_feedback
  rate_limit: 10000  # events/second
  output: "gs://lens-ml-training/cf-v2/"
```

### 2. Debugging Production Issues
**Use Case**: Investigate anomalies or errors in production

**Process**:
1. Identify time window of issue
2. Replay events with debug logging enabled
3. Capture intermediate states
4. Analyze event sequence
5. Identify root cause

**Tools**:
- Replay debugger UI in Kafka UI
- Event trace viewer
- State snapshot comparison

### 3. A/B Test Analysis
**Use Case**: Reanalyze past A/B tests with new metrics

**Process**:
1. Identify experiment ID and time range
2. Replay both control and treatment events
3. Apply new metric calculations
4. Generate updated analysis report

### 4. Compliance Audit
**Use Case**: Demonstrate data handling for regulatory audit

**Process**:
1. Replay events for specific users (with consent)
2. Generate audit trail
3. Verify data retention compliance
4. Produce audit report

### 5. Disaster Recovery
**Use Case**: Rebuild system state after catastrophic failure

**Process**:
1. Start with empty system
2. Replay all events from last known good state
3. Rebuild derived data and caches
4. Verify system consistency
5. Resume normal operations

## Implementation Details

### Event Replay Service

```python
class EventReplayService:
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.schema_registry = SchemaRegistryClient(config.registry_url)
        self.storage_client = StorageClient(config.storage_bucket)
        
    def replay_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: List[str],
        processor: EventProcessor,
        rate_limit: Optional[int] = None
    ) -> ReplayResult:
        """
        Replay events within specified time range
        """
        # Determine source (Kafka or GCS)
        source = self._determine_source(start_time)
        
        # Create replay iterator
        events = self._create_iterator(
            source, start_time, end_time, event_types
        )
        
        # Apply rate limiting if specified
        if rate_limit:
            events = self._rate_limit(events, rate_limit)
        
        # Process events
        results = ReplayResult()
        for event in events:
            try:
                # Handle schema evolution
                event = self._evolve_schema(event)
                
                # Process event
                processor.process(event)
                results.processed += 1
                
            except Exception as e:
                results.errors.append((event, e))
                
        return results
```

### State Management

During replay, system state is managed carefully:

1. **Isolation**: Replay runs in isolated environment
2. **Checkpointing**: Regular state checkpoints during long replays
3. **Idempotency**: All operations must be idempotent
4. **Versioning**: State versions tracked separately

### Monitoring and Observability

#### Metrics
- Replay progress (events/second)
- Lag behind real-time
- Error rate
- Resource utilization

#### Dashboards
- Replay progress visualization
- Error analysis
- Performance metrics
- Resource usage

### Best Practices

1. **Test Replays**: Always test replay logic in staging
2. **Rate Limiting**: Prevent overwhelming downstream systems
3. **Checkpointing**: Enable resume from failure
4. **Monitoring**: Track replay progress and errors
5. **Documentation**: Document each replay operation

## Configuration

### Replay Configuration File
```yaml
replay:
  kafka:
    bootstrap_servers: "kafka:9092"
    schema_registry: "http://schema-registry:8081"
    retention_days: 30
    
  storage:
    provider: "gcs"
    bucket: "lens-events-archive"
    format: "parquet"
    compression: "snappy"
    partition_by: "day"
    retention_years: 3
    
  processing:
    max_batch_size: 10000
    checkpoint_interval: 60  # seconds
    max_rate: 50000  # events/second
    
  monitoring:
    metrics_port: 9091
    enable_tracing: true
```

## Security Considerations

1. **Access Control**: Replay operations require special permissions
2. **Audit Logging**: All replay operations are logged
3. **Data Privacy**: PII handling during replay follows privacy policies
4. **Encryption**: Events encrypted at rest and in transit

## Operational Procedures

### Starting a Replay
1. Create replay configuration
2. Submit to replay service
3. Monitor progress
4. Verify results
5. Clean up resources

### Emergency Replay
For urgent production issues:
1. Use pre-approved replay templates
2. Bypass normal approval process
3. Notify on-call team
4. Document in incident report

## Future Enhancements

1. **ML-Powered Replay**: Intelligent event selection for replay
2. **Real-time Replay**: Replay events as they arrive with delay
3. **Federated Replay**: Replay across multiple regions
4. **Automated Testing**: Replay-based regression testing