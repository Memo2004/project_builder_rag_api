# api/main.py
from fastapi import FastAPI, Request
from rag.rag_script import generate_project  

app = FastAPI()

@app.post("/generate-project")
async def generate_project_endpoint(request: Request):
    data = await request.json()
    domain = data.get("domain")
    level = data.get("level")
    description = data.get("description")

    if not all([domain, level, description]):
        return {"error": "Missing required fields: domain, level, description"}

    response = generate_project(domain=domain, level=level, description=description)
    return response
