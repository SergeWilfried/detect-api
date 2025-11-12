# OpenAPI Specification

This directory contains the OpenAPI 3.0.3 specification for the License Plate Detection API.

## Files

- **`openapi.yaml`** - OpenAPI specification in YAML format (recommended for readability)
- **`openapi.json`** - OpenAPI specification in JSON format (for tooling compatibility)

## Usage

### Viewing the Documentation

1. **Swagger UI**: Upload `openapi.yaml` or `openapi.json` to [Swagger Editor](https://editor.swagger.io/)
2. **ReDoc**: Use [ReDoc](https://redocly.github.io/redoc/) to generate beautiful documentation
3. **FastAPI**: The API automatically generates OpenAPI docs at `/docs` and `/redoc` endpoints

### Validating the Specification

```bash
# Using Python
python -c "import yaml; yaml.safe_load(open('openapi.yaml'))"

# Using swagger-cli (if installed)
npx @apidevtools/swagger-cli validate openapi.yaml

# Using openapi-generator (if installed)
openapi-generator validate -i openapi.yaml
```

### Generating Client SDKs

You can generate client SDKs in various languages using tools like:

- **OpenAPI Generator**: https://openapi-generator.tech/
- **Swagger Codegen**: https://swagger.io/tools/swagger-codegen/

Example:
```bash
# Generate Python client
openapi-generator generate -i openapi.yaml -g python -o ./client-python

# Generate JavaScript/TypeScript client
openapi-generator generate -i openapi.yaml -g typescript-axios -o ./client-ts
```

### Importing into API Testing Tools

- **Postman**: Import `openapi.json` directly
- **Insomnia**: Import `openapi.yaml` or `openapi.json`
- **Bruno**: Import the OpenAPI spec
- **HTTPie**: Use with `--spec` flag

### Integration with FastAPI

FastAPI automatically generates OpenAPI documentation from your code. The standalone spec file is useful for:

- External documentation
- Client SDK generation
- API testing tools
- API gateway configuration
- Contract testing

To use a custom OpenAPI schema with FastAPI, you can modify `main.py`:

```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    # Load your custom schema
    import yaml
    with open("openapi.yaml") as f:
        app.openapi_schema = yaml.safe_load(f)
    
    return app.openapi_schema

app.openapi = custom_openapi
```

## API Endpoints Summary

### Health & Info
- `GET /health` - Health check
- `GET /` - API information

### Detection
- `POST /detect` - Detect from base64 or URL
- `POST /detect/upload` - Detect from file upload

### Video Processing
- `GET /detect/video` - Process frame from test video
- `POST /process/video` - Process entire video (synchronous)
- `POST /process/video/upload` - Process uploaded video (synchronous)
- `POST /process/video/upload/async` - Process uploaded video (asynchronous)

### Job Management
- `GET /jobs/{job_id}` - Get job status
- `GET /jobs/{job_id}/result` - Get job result

### Gemini AI
- `POST /gemini/detect` - Gemini object detection
- `POST /gemini/detect/upload` - Gemini detection from upload
- `POST /gemini/segment` - Gemini object segmentation
- `POST /gemini/segment/upload` - Gemini segmentation from upload

## Environment Variables

- `GEMINI_API_KEY` - Required for Gemini endpoints
- `MONGODB_URI` - Optional, for job persistence
- `REDIS_URL` - Optional, for job status caching
- `YOLO_MODEL_PATH` - Optional, custom model path
- `CONFIDENCE_THRESHOLD` - Optional, detection threshold
- `OCR_ENGINE` - Optional, 'easyocr' or 'gemini'

## Examples

See the OpenAPI spec file for detailed request/response examples for each endpoint.

