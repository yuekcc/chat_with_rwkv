import uvicorn

from . import chat
from . import server as api_server

server = api_server.app

def run():
    config = uvicorn.Config('app:server', port = 5000, log_level = 'info')
    server = uvicorn.Server(config)
    server.run()