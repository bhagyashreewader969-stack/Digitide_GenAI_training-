import os
import boto3
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))

app = FastAPI()

@app.get("/ask")
def ask(prompt: str):
    body = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": 200,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    }

    response = client.invoke_model(
        modelId='anthropic.claude-v2',
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )

    output = json.loads(response['body'].read())
    return {"response": output['completion']}
