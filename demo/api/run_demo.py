import uvicorn
import os
import sys

# Add current directory to path so we can import from .models and .utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting NER Demo API...")
    print("Frontend UI: http://127.0.0.1:8000")
    print("Swagger UI will be available at: http://localhost:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
