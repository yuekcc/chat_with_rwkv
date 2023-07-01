from io import StringIO

from fastapi import FastAPI

from app import chat

app = FastAPI()

@app.get('/')
async def handle_home():
    return { "message": "hello, world" }

@app.get('/api/v1/chat/{session_id}')
async def handle_chat(session_id, query):
    session = chat.get_session(session_id)
    words = session.eval(query)

    buf = StringIO()
    for word in words:
        buf.write(word)

    buf.flush()
    buf.seek(0)
    content = buf.read()
    return {
        "content": content
    }
