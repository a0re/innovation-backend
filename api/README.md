# Spam Detection API

Clean, modular FastAPI application for spam detection with production-ready features.

## Architecture

The API follows a clean, modular architecture with separation of concerns:

```
api/
├── main.py                 # Application entry point (232 lines)
├── middleware.py           # Rate limiting & logging middleware
├── dependencies.py         # Dependency injection (API key, model)
├── schemas.py              # Pydantic models for validation
├── error_handlers.py       # Exception handlers
├── database.py             # Database operations
├── routes/                 # Route handlers by feature
│   ├── predictions.py      # Prediction endpoints
│   ├── general.py          # Health, info, examples
│   ├── statistics.py       # Stats and history
│   └── clustering.py       # Clustering endpoints
├── requirements.txt        # API dependencies
└── .env                    # Configuration
```

## Key Features

### Security
- **API Key Authentication**: Optional X-API-Key header validation
- **Rate Limiting**: IP-based rate limiting (configurable)
- **CORS Protection**: Whitelist allowed origins
- **Input Validation**: Pydantic models with custom validators

### Performance
- **GZip Compression**: Automatic response compression
- **Database Optimization**: Single-query statistics
- **Async/Await**: Non-blocking I/O operations

### Code Quality
- **Modular Design**: Separated concerns into focused modules
- **Dependency Injection**: Functions injected at route creation
- **Type Hints**: Full type annotations throughout
- **Clean Structure**: ~70-80 lines per route module

## Configuration

All configuration via environment variables in `.env`:

```bash
# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Security
API_KEY=your-secret-key  # Optional, leave empty to disable
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# CORS
CORS_ORIGINS=http://localhost:5174,http://localhost:5173
```

## Running the API

```bash
# Development
python api/main.py

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Predictions
- `POST /predict` - Single prediction (best model)
- `POST /predict/batch` - Batch predictions
- `POST /predict/multi-model` - All models + ensemble
- `POST /predict/multi-model/batch` - Batch multi-model

### General
- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `GET /examples` - Sample messages

### Statistics
- `GET /stats` - Prediction statistics
- `DELETE /stats` - Reset statistics
- `GET /predictions/recent` - Recent predictions
- `GET /predictions/{id}` - Specific prediction
- `PUT /predictions/{id}/feedback` - Update feedback
- `GET /predictions/date-range` - Date range query
- `DELETE /predictions/old` - Delete old data

### Clustering
- `GET /cluster/info` - Cluster information

## Module Documentation

### `main.py`
Clean entry point that:
- Loads environment configuration
- Initializes FastAPI with lifespan context
- Registers middleware (CORS, GZip, rate limiting)
- Creates and registers route handlers
- Sets up error handlers

### `middleware.py`
Middleware components:
- `RateLimiter` - Class-based rate limiter with IP tracking
- `log_request_middleware` - Request timing logger

### `dependencies.py`
Dependency factory functions:
- `create_api_key_validator()` - Returns API key validator
- `create_model_dependency()` - Returns model availability checker

### `schemas.py`
Pydantic models with validation:
- Request models with field validators
- Response models with type safety
- Nested models for complex responses

### `error_handlers.py`
Exception handlers:
- HTTP exceptions with consistent format
- Validation errors
- General exception catchall

### `routes/`
Feature-based route modules:
- Each module has a factory function
- Dependencies injected at creation time
- Returns configured APIRouter
- Clean, focused endpoint handlers

## Design Principles

1. **Separation of Concerns**: Each file has a single responsibility
2. **Dependency Injection**: Routes receive dependencies, not imports
3. **Factory Pattern**: Route creators allow flexible configuration
4. **Type Safety**: Full type hints for IDE support
5. **Error Handling**: Consistent error responses
6. **Testability**: Easy to test with dependency injection
7. **Maintainability**: Small, focused modules

## Benefits of This Architecture

- ✅ **Reduced main.py**: From ~850 lines to ~232 lines
- ✅ **Modularity**: Easy to find and modify specific features
- ✅ **Reusability**: Middleware and dependencies can be reused
- ✅ **Testability**: Each module can be tested independently
- ✅ **Scalability**: Easy to add new routes or features
- ✅ **Maintainability**: Clear structure, easy onboarding

## Adding New Endpoints

1. Create a new route module in `routes/`
2. Define a factory function that takes dependencies
3. Register the router in `main.py`

Example:

```python
# routes/new_feature.py
from fastapi import APIRouter

def create_new_feature_routes(dependency):
    router = APIRouter(tags=["NewFeature"])

    @router.get("/new-endpoint")
    async def new_endpoint():
        return {"message": "Hello"}

    return router

# main.py
from routes.new_feature import create_new_feature_routes

new_router = create_new_feature_routes(dependency=some_value)
app.include_router(new_router)
```

## Performance Notes

- Rate limiter uses in-memory storage (consider Redis for production scaling)
- Database uses SQLite (consider PostgreSQL for production)
- Single model instance loaded at startup (shared across requests)
- GZip compression reduces response size by ~60-80%

## Security Notes

- API key validation is optional but recommended
- Rate limiting prevents abuse and DoS
- CORS restricts origins
- All inputs validated with Pydantic
- Error messages don't leak sensitive info
