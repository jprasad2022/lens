# Understanding the LENS Codebase

This guide will help you understand how the LENS movie recommendation system works, step by step.

## 🎯 What Does LENS Do?

LENS is a movie recommendation system that:
1. **Shows movies** to users based on their preferences
2. **Learns** from user interactions (clicks, ratings)
3. **Improves** recommendations over time using machine learning
4. **Monitors** its own performance

Think of it like Netflix's recommendation system but simpler and educational.

## 🏗️ System Architecture - The Big Picture

```
User → Frontend → Backend API → ML Models → Recommendations
         ↓           ↓              ↓
    [React App]  [FastAPI]    [Scikit-learn]
                     ↓
                  Database
                  (Movies)
```

## 📁 Key Directories Explained

### 1. **Frontend** (`/frontend`) - What Users See
- **Purpose**: The website users interact with
- **Technology**: Next.js (React framework)
- **Key Files**:
  - `src/app/page.js` - Homepage
  - `src/components/recommendations/RecommendationGrid.jsx` - Shows movie cards
  - `src/services/api.service.js` - Talks to backend

**Example Flow**: User visits website → Types user ID → Clicks "Get Recommendations" → Sees movies

### 2. **Backend** (`/backend`) - The Brain
- **Purpose**: Processes requests and runs ML models
- **Technology**: FastAPI (Python web framework)
- **Key Files**:
  - `app/main.py` - Entry point, starts the server
  - `routers/recommendation_router.py` - API endpoints
  - `services/recommendation_service.py` - Business logic
  - `recommender/models.py` - ML algorithms

**Example Flow**: Receives user ID → Loads ML model → Calculates recommendations → Returns movie list

### 3. **Schemas** (`/schemas`) - Data Contracts
- **Purpose**: Defines the structure of messages between services
- **Technology**: Avro schemas for Kafka
- **Files**: Each `.avsc` file describes a message type

## 🔄 How Data Flows Through the System

### Simple Recommendation Flow:
```
1. User opens website (Frontend)
   ↓
2. User enters their ID and clicks "Get Recommendations"
   ↓
3. Frontend sends request to Backend API
   GET http://localhost:8000/recommend/123?k=10
   ↓
4. Backend receives request
   ↓
5. Backend loads the ML model (e.g., collaborative filtering)
   ↓
6. Model calculates top 10 movies for user 123
   ↓
7. Backend returns movie list with details
   ↓
8. Frontend displays movies as cards
```

### Real-time Learning Flow (with Kafka):
```
1. User rates a movie
   ↓
2. Frontend sends rating to Backend
   ↓
3. Backend publishes event to Kafka
   Topic: "user-interactions"
   Message: {userId: 123, movieId: 456, rating: 5}
   ↓
4. Stream processor consumes event
   ↓
5. Updates model with new data
   ↓
6. Future recommendations improve
```

## 🧩 Key Components Explained

### Backend Components

1. **Models** (`backend/recommender/models.py`)
   - **Popularity Model**: Recommends most popular movies
   - **Collaborative Filtering**: Finds similar users
   - **ALS (Alternating Least Squares)**: Matrix factorization

2. **Services** (`backend/services/`)
   - `recommendation_service.py`: Orchestrates the recommendation process
   - `model_service.py`: Loads and manages ML models
   - `cache_service.py`: Speeds up repeated requests
   - `kafka_service.py`: Handles real-time events

3. **Routers** (`backend/routers/`)
   - Define API endpoints
   - Handle HTTP requests/responses
   - Validate input data

### Frontend Components

1. **Pages** (`frontend/src/app/`)
   - `page.js`: Main recommendation interface
   - `monitoring/page.js`: Performance metrics
   - `about/page.js`: Information page

2. **Components** (`frontend/src/components/`)
   - `RecommendationCard.jsx`: Single movie display
   - `SearchBar.js`: User input interface
   - `ModelSelector.jsx`: Choose ML algorithm

## 🚀 Common Tasks & Where to Look

### "I want to add a new ML model"
1. Add model class in `backend/recommender/models.py`
2. Register it in `backend/services/model_service.py`
3. Add to frontend dropdown in `frontend/src/components/ui/ModelSelector.jsx`

### "I want to change the UI design"
1. Modify components in `frontend/src/components/`
2. Update styles in `frontend/src/app/globals.css`
3. Use Tailwind CSS classes

### "I want to add a new API endpoint"
1. Create function in `backend/routers/recommendation_router.py`
2. Add business logic in `backend/services/`
3. Update frontend API calls in `frontend/src/services/api.service.js`

### "I want to track a new metric"
1. Add metric in backend code using Prometheus
2. Update `infra/prometheus.yml` if needed
3. Create Grafana dashboard

## 🔍 Understanding the Code - Start Here

### Step 1: Trace a Simple Request
1. Open `frontend/src/app/page.js`
2. Find the "Get Recommendations" button
3. Follow the `handleSubmit` function
4. See how it calls the API
5. Open `backend/routers/recommendation_router.py`
6. Find the matching endpoint
7. Follow the code flow

### Step 2: Understand Data Models
1. Look at `backend/models/schemas.py`
2. See how data is structured
3. Match with frontend types

### Step 3: Explore ML Models
1. Open `backend/recommender/models.py`
2. Start with `PopularityRecommender` (simplest)
3. Then look at `CollaborativeFilteringRecommender`
4. Finally `ALSRecommender` (most complex)

## 📊 Services & Their Purposes

| Service | Port | Purpose | When to Check |
|---------|------|---------|---------------|
| Frontend | 3000 | User interface | UI issues |
| Backend | 8000 | API & ML models | Logic errors |
| Kafka | 9092 | Event streaming | Real-time features |
| Redis | 6379 | Caching | Performance |
| Prometheus | 9090 | Metrics collection | Monitoring |
| Grafana | 3001 | Visualization | Dashboards |

## 🐛 Debugging Tips

### Frontend Issues:
- Open browser DevTools (F12)
- Check Network tab for API calls
- Look at Console for errors

### Backend Issues:
- Check terminal logs
- Visit http://localhost:8000/docs
- Test API endpoints directly

### Docker Issues:
- Run `docker-compose logs [service-name]`
- Check if all containers are running: `docker-compose ps`

## 📚 Learning Path

1. **Start Simple**: 
   - Run the app locally
   - Make a small UI change
   - See it reflected

2. **Explore API**:
   - Visit http://localhost:8000/docs
   - Try different endpoints
   - Understand request/response

3. **Modify Logic**:
   - Change recommendation count
   - Add logging
   - Filter results

4. **Advanced**:
   - Add new model
   - Implement caching
   - Create metrics

## 💡 Key Concepts to Understand

1. **REST API**: How frontend and backend communicate
2. **Machine Learning Models**: How recommendations are calculated
3. **Event Streaming**: How real-time updates work
4. **Containerization**: How services run independently
5. **Monitoring**: How we track system health

## 🎓 Next Steps

1. Pick one component (e.g., Frontend or Backend)
2. Read its README file
3. Make a small change
4. Test it works
5. Gradually explore more components

Remember: You don't need to understand everything at once. Start with what interests you most!