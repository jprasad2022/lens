# MovieLens Frontend

A configurable and modular frontend for the MovieLens recommendation system, following the architectural patterns from the insurance-rag-app.

## Architecture Overview

This frontend follows a modern, modular architecture with:

- **Next.js 14** with App Router
- **React Query** for server state management
- **Context API** for global state
- **Tailwind CSS** for styling
- **Service Layer** for API interactions
- **Configurable Features** via environment variables

## Project Structure

```
src/
├── app/                    # Next.js app router pages
│   ├── page.js            # Home page
│   ├── about/             # About page
│   ├── search/            # Movie search page
│   ├── monitoring/        # Monitoring dashboard
│   ├── experiments/       # A/B testing dashboard
│   ├── layout.js          # Root layout
│   ├── providers.jsx      # App providers
│   └── globals.css        # Global styles
├── components/            # Reusable components
│   ├── recommendations/   # Recommendation display components
│   │   ├── RecommendationCard.jsx
│   │   └── RecommendationGrid.jsx
│   ├── search/           # Search components
│   │   ├── MovieCard.js   # Movie display card
│   │   ├── Rating.js      # Rating component
│   │   ├── SearchBar.js   # Search input
│   │   └── SearchResults.js # Search results display
│   ├── monitoring/        # Metrics and monitoring components
│   │   ├── ABTestResults.jsx
│   │   ├── LatencyChart.jsx
│   │   ├── MetricCard.jsx
│   │   ├── MetricsDashboard.jsx
│   │   └── SystemHealth.jsx
│   ├── ui/               # Generic UI components
│   │   ├── ErrorMessage.jsx
│   │   ├── LoadingSpinner.jsx
│   │   ├── ModelSelector.jsx
│   │   └── UserInput.jsx
│   └── layout/           # Layout components
│       └── Header.jsx
├── config/               # Configuration management
│   └── app.config.js     # Centralized app configuration
├── contexts/             # React contexts
│   ├── AuthContext.jsx   # Authentication state
│   └── AppContext.jsx    # Application state
├── services/             # Service layer
│   ├── api.service.js    # API client with retry logic
│   ├── auth.service.js   # Authentication service
│   └── movie.service.js  # Movie data service
├── hooks/                # Custom React hooks
│   └── useMovie.js       # Movie data hook
├── lib/                  # Library utilities
├── types/                # TypeScript types (if using TS)
└── utils/                # Utility functions
```

## Configuration

### Environment Variables

Create a `.env.local` file:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_KAFKA_ENABLED=true

# Firebase Configuration (Optional)
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=

# Feature Flags
NEXT_PUBLIC_ENABLE_A_B_TESTING=true
NEXT_PUBLIC_ENABLE_MONITORING=true
NEXT_PUBLIC_ENABLE_OFFLINE_EVAL=true

# Model Configuration
NEXT_PUBLIC_DEFAULT_MODEL=popularity
NEXT_PUBLIC_RECOMMENDATION_COUNT=20
```

### Centralized Configuration

All configuration is managed through `src/config/app.config.js`, which provides:

- API endpoints and settings
- Feature flags
- UI preferences
- Model configuration
- Monitoring settings

## Key Features

### 1. Modular Service Layer

```javascript
// API calls with automatic retry and error handling
const recommendations = await apiService.getRecommendations(userId, {
  k: 20,
  model: 'collaborative'
});
```

### 2. Context-Based State Management

- **AuthContext**: Handles user authentication (optional)
- **AppContext**: Manages global app state (models, metrics, health)

### 3. Real-time Monitoring

- Prometheus metrics integration
- Live system health checks
- Performance dashboards
- A/B test results visualization

### 4. Configurable Features

Enable/disable features via environment variables:
- Authentication (Firebase)
- A/B Testing
- Monitoring dashboards
- Offline evaluation tools

### 5. Error Handling & Retry Logic

- Automatic retry for 5xx errors
- Exponential backoff
- User-friendly error messages
- Network error detection

## Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## API Integration

The frontend expects the backend API to provide:

```
GET /recommend/{user_id}?k=20&model=popularity
GET /healthz
GET /metrics
POST /feedback
GET /models
POST /switch
```

## Deployment

### Development
```bash
npm run dev
```

### Production Build
```bash
npm run build
npm start
```

### Docker
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package*.json ./
RUN npm ci --production
EXPOSE 3000
CMD ["npm", "start"]
```

## Security Features

- Secure headers configuration
- CSRF protection
- Environment variable validation
- Authentication token handling
- Input sanitization

## Performance Optimizations

- React Query caching
- Lazy loading for Firebase
- Image optimization with Next.js
- API response caching
- Debounced search inputs

## Testing

```bash
# Run tests
npm test

# Run tests in CI mode
npm run test:ci
```

## Contributing

1. Follow the existing patterns for new components
2. Use the service layer for all API calls
3. Add new features behind feature flags
4. Update configuration documentation
5. Write tests for new components

## License

MIT