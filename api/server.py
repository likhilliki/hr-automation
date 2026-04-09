from fastapi import FastAPI, Body, Query
from fastapi.responses import JSONResponse
from env.environment import HREnvironment
from env.models import Action, Observation, Reward

app = FastAPI(title="HR Hiring Agent Environment")

env = HREnvironment(task_id="easy")

@app.get("/reset", response_model=Observation)
def reset(task_id: str | None = Query(None, description="Optional task identifier: easy, medium, hard")):
    observation = env.reset(task_id)
    return observation

@app.post("/step")
def step(action: Action = Body(...)):
    observation, reward, done, info = env.step(action.dict())
    payload = {
        "observation": observation.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }
    return JSONResponse(content=payload)

@app.get("/state")
def state():
    return env.state()
