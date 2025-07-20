# Wound Classification Model

A FastAPI-based web application for classifying wound types using a PyTorch ResNet18 model.

## Features

- **Wound Classification**: Classifies wounds into 6 categories (Eschar, Granulating Tissue, Healthy Tissue, Necrotic Tissue, Slough, Undefined)
- **Urgency Assessment**: Provides urgency levels and hospital visit recommendations
- **Web Interface**: User-friendly web UI for image upload and classification
- **REST API**: Full API documentation with Swagger UI
- **Health Monitoring**: Built-in health checks for container orchestration

## Docker Setup

### Prerequisites

- Docker
- Docker Compose

### Quick Start

1. **Build and run with Docker Compose:**

   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Web Interface: http://localhost:8080
   - API Documentation: http://localhost:8080/docs
   - Health Check: http://localhost:8080/health

### Manual Docker Build

1. **Build the image:**

   ```bash
   docker build -t wound-classifier .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 wound-classifier
   ```

## API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

### Predict Wound Type

```bash
curl -X POST "http://localhost:8080/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

### API Information

```bash
curl http://localhost:8080/api-info
```

## Model Information

- **Architecture**: ResNet18 with custom classifier head
- **Input**: RGB images (224x224 pixels)
- **Output**: 6-class classification with confidence scores
- **Model File**: `model/model.pth` (83MB)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Response Format

```json
{
  "predicted_class": "Granulating Tissue",
  "confidence": 0.85,
  "urgency_level": "LOW",
  "requires_hospital": false,
  "recommendations": [
    "Continue prescribed wound care",
    "Keep the area clean and moist",
    "Monitor for signs of infection",
    "Follow up with healthcare provider as scheduled"
  ],
  "class_probabilities": {
    "Granulating Tissue": 0.85,
    "Healthy Tissue": 0.1,
    "Slough": 0.03,
    "Undefined": 0.01,
    "Eschar": 0.005,
    "Necrotic Tissue": 0.005
  }
}
```

## Development

### Local Setup (without Docker)

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8080
   ```

### Environment Variables

- `PYTHONPATH`: Set to `/app` in Docker
- `PYTHONUNBUFFERED`: Set to `1` for immediate log output

## Docker Configuration

- **Base Image**: Python 3.9-slim
- **Port**: 8080
- **Memory Limit**: 2GB
- **Health Check**: Every 30 seconds
- **Restart Policy**: Unless stopped

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `model/model.pth` exists in the project directory
2. **Port already in use**: Change the port mapping in `docker-compose.yml`
3. **Memory issues**: Increase memory limits in Docker settings

### Logs

View container logs:

```bash
docker-compose logs -f wound-classifier
```

### Health Check

The application includes a health check endpoint that verifies:

- Server is running
- Model is loaded
- Device configuration (CPU/GPU)
