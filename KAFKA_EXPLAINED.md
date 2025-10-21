# When and Why Kafka Comes Into Play in LENS

## 🤔 What is Kafka?

Think of Kafka as a **message delivery system** - like a post office for your application:
- Apps send messages (events) to Kafka
- Kafka stores and delivers them to interested apps
- Messages don't get lost even if the receiver is temporarily offline

## 📍 When Does Kafka Get Used in LENS?

### 1. **User Interactions Tracking**
When a user interacts with the system, events are sent to Kafka:

```
User clicks on a movie → Frontend → Backend → Kafka Topic: "user-interactions"
User rates a movie → Frontend → Backend → Kafka Topic: "user-interactions"
User bookmarks a movie → Frontend → Backend → Kafka Topic: "user-interactions"
```

**Why not just save to database?**
- Kafka allows multiple systems to react to the same event
- Events are processed asynchronously (doesn't slow down the user)
- Events can be replayed if needed

### 2. **Recommendation Requests Logging**
Every time someone asks for recommendations:

```
User requests recommendations → Backend → Kafka Topic: "recommendation-requests"
```

This helps track:
- Which users are active
- Which models are being used
- Peak usage times

### 3. **Recommendation Responses Tracking**
When the system returns recommendations:

```
Backend generates recommendations → Kafka Topic: "recommendation-responses"
```

This captures:
- What was recommended
- How long it took
- Whether it succeeded

### 4. **Model Performance Metrics**
ML models report their performance:

```
Model calculates accuracy → Kafka Topic: "model-metrics"
Model measures latency → Kafka Topic: "model-metrics"
```

### 5. **A/B Testing Events**
When running experiments:

```
User assigned to test group → Kafka Topic: "ab-test-events"
User converts (clicks/rates) → Kafka Topic: "ab-test-events"
```

## 🔄 The Kafka Flow in LENS

```
1. User rates a movie 5 stars
   ↓
2. Frontend sends POST request to Backend
   ↓
3. Backend saves to database AND sends to Kafka
   ↓
4. Kafka stores message in "user-interactions" topic
   ↓
5. Multiple consumers can read this event:
   - Analytics service → Updates user statistics
   - ML training service → Retrains models
   - Monitoring service → Tracks engagement
   - Recommendation cache → Invalidates old recommendations
```

## 🏗️ How It's Implemented

### Backend Publishes Events

```python
# In backend/services/recommendation_service.py

# When user interacts with a movie
async def track_interaction(user_id, movie_id, interaction_type):
    # Save to database
    save_to_db(...)
    
    # Also send to Kafka
    event = {
        "userId": user_id,
        "movieId": movie_id,
        "interactionType": interaction_type,
        "timestamp": time.time()
    }
    await kafka_service.publish("user-interactions", event)
```

### Stream Processor Consumes Events

```python
# In backend/stream/consumer.py

# Continuously reads from Kafka
async def process_events():
    while True:
        message = await kafka.consume("user-interactions")
        
        # Update user profile
        update_user_preferences(message)
        
        # Trigger model retraining if needed
        if should_retrain():
            retrain_model()
        
        # Update metrics
        update_engagement_metrics(message)
```

## 📊 Real Examples in LENS

### Example 1: Real-time Model Updates
```
1. 100 users rate movies throughout the day
2. Each rating goes to Kafka
3. Every hour, a job reads all ratings from Kafka
4. Updates the collaborative filtering model
5. Next user gets better recommendations
```

### Example 2: Performance Monitoring
```
1. Every recommendation request is logged to Kafka
2. Monitoring service reads these events
3. Calculates metrics like:
   - Average response time
   - Success rate
   - Popular models
4. Displays in Grafana dashboards
```

### Example 3: User Behavior Analysis
```
1. User searches for "action movies"
2. Clicks on "Die Hard"
3. Rates it 4 stars
4. All three events go to Kafka
5. Analytics service builds user journey
6. Identifies that searches lead to higher ratings
```

## 🚀 When to Use Kafka vs Direct Database

### Use Kafka When:
- Multiple services need the same information
- You want to process events asynchronously
- Order of events matters
- You need event replay capability
- Building real-time features

### Use Direct Database When:
- Simple CRUD operations
- Only one service needs the data
- Immediate consistency required
- Querying historical data

## 🛠️ Kafka Components in LENS

1. **Zookeeper** (Port 2181)
   - Manages Kafka cluster
   - Tracks topic metadata

2. **Kafka Broker** (Port 9092)
   - Stores messages
   - Handles publish/subscribe

3. **Schema Registry** (Port 8081)
   - Stores message schemas
   - Ensures compatibility

4. **Kafka UI** (Port 8080)
   - Visual interface
   - Browse topics and messages

## 🔍 How to See Kafka in Action

1. **View Topics in Kafka UI**:
   - Go to http://localhost:8080
   - Click on "Topics"
   - See all message types

2. **Publish a Test Event**:
   ```bash
   # In the backend directory
   python -c "
   from services.kafka_service import KafkaService
   import asyncio
   
   async def test():
       kafka = KafkaService()
       await kafka.publish('user-interactions', {
           'userId': 'test123',
           'movieId': '1',
           'interactionType': 'view',
           'timestamp': 1234567890
       })
   
   asyncio.run(test())
   "
   ```

3. **Watch Events Flow**:
   - Make interactions in the frontend
   - Check Kafka UI for new messages
   - See them appear in real-time

## 💡 Key Benefits for LENS

1. **Scalability**: Can handle millions of events
2. **Reliability**: Events aren't lost if a service crashes
3. **Flexibility**: Easy to add new consumers
4. **Real-time**: Enables live updates
5. **Decoupling**: Services don't need to know about each other

## 🎯 Without Kafka vs With Kafka

### Without Kafka:
```
User rates movie → Backend → Database
                          ↓
                    Update model (blocking)
                          ↓
                    Return response (slow)
```

### With Kafka:
```
User rates movie → Backend → Database
                          ↘
                           Kafka (async) → Update model
                          ↙              ↘
                Return response (fast)     Analytics
                                         ↘
                                          Monitoring
```

The user gets a fast response while heavy processing happens in the background!