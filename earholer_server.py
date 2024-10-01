import asyncio
import zmq
import msgpack
from capture_filter import MessageFilter
from config import *
import json

logging.basicConfig(
    level=SEVER_LOG_LEVEL,
    format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class Earholer():
    
    harraq_filter = MessageFilter()
    
    def __init__(self, cf) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://127.0.0.1:{EARHOLER_PORT}")
        self.cf = cf
    
    def handle_message(self, message):
        try:
            logger.info(message)
            text = message['text']
            if message['type'] == 'ping':
                return ({'type': 'pong'})
            gen_dot_douglas = self.cf(text)
            if gen_dot_douglas is None:
                return "NOTTE"
            return gen_dot_douglas
        except Exception as e:
            logger.error(e)
            return "HUUNGNANHJANFHAISFJ ASNFUASH WASGW FAM WHA WKD"

    def run(self):
        logger.info("EARHOLER RUNNY")
        while True:
            packed_message = None
            try:
                packed_message = self._socket.recv_string()
                message = json.loads(packed_message)
                response = self.handle_message(message)
                packed_response = json.dumps(response)
                self._socket.send_string(packed_response)
            except Exception as e:
                if "current state" in str(e):
                    self._socket.close()
                    self._socket = self._context.socket(zmq.REP)
                    self._socket.bind(f"tcp://127.0.0.1:{EARHOLER_PORT}")
                logger.error(f"Error processing message: {str(e)}")
                logger.error(packed_message)
                
            
