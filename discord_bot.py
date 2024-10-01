import re
import discord
from discord.ext import tasks
import sqlite3
import logging
import time
import zmq
import msgpack
import random
import json
from config import *
from capture_filter import MessageFilter
import logging

logging.basicConfig(
    level=logging.ERROR,
    format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


conn = sqlite3.connect('./db/optout.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
table_exists = cursor.fetchone()
if not table_exists:
    cursor.execute('''CREATE TABLE users
                      (user_id INTEGER PRIMARY KEY, username TEXT, nickname TEXT)''')
    conn.commit()

class Context():
    index = 0
    size = 250
    chat_context = None
    last = "Mubaraq"
    count = 0
    k = 6
    word_array = ["cp", "nigger", "nigga", "child", "children", "pedo", "pedophillia", "epstein", "kkk"]

    def __init__(self, size):
        self.size = size
        self.chat_context = [None] * size

    def append(self, item):
        self.last = item
        if item not in self.chat_context:
            self.chat_context[self.index] = item
            self.index = (self.index + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def get(self):
        return [x for x in self.chat_context if x is not None]

    def is_blocked(self, string):
        result = string
        for word in self.word_array:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            result = pattern.sub('', result)
        return result
    
    def to_string(self):
        items = self.get()
        if self.count > 1:
            # logger.info("HERE ALZO")
            last_index = (self.index - 1) % self.size
            weights = [2 ** i for i in range(self.count)]
            items = [items[i] for i in random.choices(
                range(self.count), weights=weights, k=min(self.count, 3))]
            items.append(self.last)
            new_array = [b for b in items if len(b) <= 4*len(self.last)]
            # logger.info("=====")
            # logger.info(self.context)
            # logger.info(items)
            # logger.info(new_array)
            # logger.info("=====")
            return ". ".join(list(set([item.capitalize() for item in new_array])))
        else:
            return self.last

class DiscordClient(discord.Client):

    muhharaq = "business on the business"
    badka = []
    chat_context = {}
    pipes = {}
    temp = 1.2
    beam = 4

    def __init__(self, intents):
        super().__init__(intents=intents)
        self.ready = False
        self.timeout_duration = 60*60*1
        self.message_channel = None
        self.last_message_time = discord.utils.utcnow()
        self.timer_started = False
        self.top_layer = False
        self.markov_layer = False
        
        self.patterns = {
            r'<#\d+>': 'DISCORD_CHANNEL',
            r'<@\d+>': 'DISCORD_MENTION',
            r'https?://\S+': 'URL',
            r'<:\w+:\d+>': 'DISCORD_EMOJI'
        }
        self.markov_context = zmq.Context()
        self.markov_socket = self.markov_context.socket(zmq.REQ)
        self.markov_socket.connect(f"tcp://127.0.0.1:{MARKOV_PORT}")
        #self.markov_socket.setsockopt(zmq.RCVTIMEO, 10000)

        self.top_layer_context = zmq.Context()
        self.top_layer_socket = self.top_layer_context.socket(zmq.REQ)
        self.top_layer_socket.connect(f"tcp://127.0.0.1:{TOP_LAYER_PORT}")
        

    async def on_thread_join(self, thread):
        await thread.join()
        await thread.send(f"wecloomes mysekf")
        
    async def on_ready(self):
        print(" ===========================================AM READINGSONSS")
        self.ready = True
        self.timer_task.start()
        self.heartbeat_task.start()
        if USERPHONE_CHANNEL:
            await self.get_channel(USERPHONE_CHANNEL).send("HUHHARABIN")
            await self.userphon(self.get_channel(USERPHONE_CHANNEL))

        
        
        
        
    # task  o
    async def userphon(self, channel):
        # Update the last message time and reset the timer
        self.last_message_time = discord.utils.utcnow()
        
        if not self.timer_started:
            self.message_channel = channel
            self.timer_started = True
            self.timer_task.restart()

    @tasks.loop(seconds=10)
    async def timer_task(self):
        if self.last_message_time is None:
            return
        
        elapsed_time = (discord.utils.utcnow() - self.last_message_time).total_seconds()
        if elapsed_time >= self.timeout_duration:
            try:
                if self.message_channel and USERPHONE_CHANNEL:
                    await self.get_channel(USERPHONE_CHANNEL).send("--userphone")
                self.last_message_time = discord.utils.utcnow()  # Reset the timer
            except Exception as e:
                logger.error(str(e))

    @timer_task.before_loop
    async def before_timer_task(self):
        await self.wait_until_ready()
    
    @tasks.loop(seconds=60)
    async def heartbeat_task(self):
        try:
            message = {
                "text": "hiran",
                "type": "ping"
            }
            
            self.top_layer_socket.setsockopt(zmq.RCVTIMEO, 500)
            response = self.pack_and_send(self.top_layer_socket, message)
            self.top_layer_socket.setsockopt(zmq.RCVTIMEO, 60000)
            self.top_layer = True
            logger.info(response)
        except:
            self.top_layer = False
            logger.error("ANTI HEART BEAT TOP LAYER")
            self.top_layer_socket.close()
            self.top_layer_socket = self.top_layer_context.socket(zmq.REQ)
            self.top_layer_socket.connect(f"tcp://127.0.0.1:{TOP_LAYER_PORT}")
        
        try:
            message = {
                "text": "hiran",
                "type": "ping"
            }
            
            self.markov_socket.setsockopt(zmq.RCVTIMEO, 500)
            response = self.pack_and_send(self.markov_socket, message)
            self.markov_socket.setsockopt(zmq.RCVTIMEO, 60000)
            self.markov_layer = True
            logger.info(response)
        except:
            self.markov_layer = False
            logger.error("ANTI HEARTBEAT MARKOV")
            self.markov_socket.close()
            self.markov_socket = self.top_layer_context.socket(zmq.REQ)
            self.markov_socket.connect(f"tcp://127.0.0.1:{MARKOV_PORT}")
            

        
        
        
        
    # utilities o
    def pack_and_send(self, socket, data):
        packed_data = msgpack.packb(data)
        socket.send(packed_data)
        packed_response = socket.recv()
        response = msgpack.unpackb(packed_response)
        return response
    
    def filter_mentions(self, reply: str):
        try:
            huhharchkinson = re.findall(r'@\d{18}', reply)
            for TURRTURRR in huhharchkinson:
                user_id = int(TURRTURRR.replace("@", ""))
                cursor.execute("SELECT nickname FROM users WHERE user_id=?", (user_id,))
                result = cursor.fetchone()
                if result is not None:
                    nickname = result[0]
                    reply = reply.replace(TURRTURRR, nickname)
            return reply
        except Exception as e:
            logger.error(e)
            return reply
        
    def tag_patterns(self, text):
        placeholders = {}  # Store placeholders and replacements
        cnt = 0
        for pattern, placeholder in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                placeholder_with_id = f'{placeholder}_{len(placeholders)}'
                text = text.replace(match.group(0), placeholder_with_id)
                placeholders[placeholder_with_id] = match.group(0)
        return text, placeholders

    def untag_patterns(self, text, placeholders):
        for placeholder, original_text in placeholders.items():
            text = text.replace(placeholder, original_text)
        return text

    def pipeka_numberka(self, hubinta: str) -> int:
        baaritanka = re.fullmatch(r'pipe(\d+)', hubinta)
        if baaritanka:
            numbet = int(baaritanka.group(1))
            return numbet
        else:
            return None

    def filterka(self, texty: str) -> str:
        reply = re.sub(BLOCK_PHRASE_PRIVACY, '', texty, flags=re.IGNORECASE)
        reply = re.sub(r' +', ' ', reply)
        reply = reply.strip()
        if reply == '':
            reply = "Whatka mm m m m m m m mmm m?"
        return reply
    
    def get_pipe(self, chan_idka):
        pipe_ = 0
        if chan_idka in self.pipes:
            pipe_ = self.pipes[chan_idka]
        return pipe_
    
    
    
    
    # commandka 
    async def voice_commands(self, message):
        if message.content.startswith("voicekajoinka") and str(message.author.id) in ['345018122924982276', '142705172395589632', '271323129987465216', "921943732775964673"]:
            self.currentChannel = message
            channel = message.author.voice.channel
            try:
                await channel.connect()
            except Exception as e:
                logger.error(e)
                await message.channel.send("shit broo exception")
        if message.content.startswith("fuckaoffka") and str(message.author.id) in ['345018122924982276', '142705172395589632', '271323129987465216', "921943732775964673"]:
            if message.guild.voice_client:
                await message.guild.voice_client.disconnect()
            else:
                await message.channel.send("shit broo")
        return
    
    async def processka_comandkionson(self, message: discord.Message):
       #antti anti opt outka
        if message.content.startswith('tarrachka tarrachka anti say me my laxanka number'):
            CHAEHCCHAECH = message.author.display_name
            cursor.execute("INSERT INTO users (user_id, username, nickname) VALUES (?, ?, ?)",
                        (message.author.id, message.author.name, CHAEHCCHAECH))
            conn.commit()
            msg = ConnectorRecvMessage("REMOVKINSONS AM REMOVE YOU AM REMOVKINSONS AM REMOVED YOUSKA AM REMOVED YOU OKAY AM REMOVED YOUR AM NOT AM NOT AM YES IYESIYESIYES I YES DIDKINSON MR NEVISHI AM YES YES So, regarding your silly ping...  Did you consider that I'm a grown-ass man, with a life?   That perhaps your joke wouldn't be very funny to me? Granted, not much of a life, but I had to pause my youtube video, switch to discord, read back several pages to see what the deal was.")
            reply = self.gen_0(msg, message, False, True, False)
            await message.channel.send(f"IYES ~~~~~ ~ ~ {reply}")

        #opt outka (antI)
        if message.content.startswith('tarrachka tarrachka say me my laxanka number'):
            cursor.execute("DELETE FROM users WHERE user_id=?", (message.author.id,))
            conn.commit()
            msg = ConnectorRecvMessage("i will pings you mr  anushi i will pings i will pingkinsosn like a pinkie finger")
            reply = self.gen_0(msg, message, False, True, False)
            await message.channel.send(f"I MOT !!!!11 ```cool 111```{reply}")
            
        # tekob anti reply gen channelka
        if 'clearkalistka' in message.content:
            self.badka = []
            
        #specially pipeline
        pipe_numbet = self.pipeka_numberka(message.content)
        if pipe_numbet:
            self.pipes[message.channel.id] = pipe_numbet

        # temnmpka
        if message.content.startswith('temp '):
            try:
                temp_value = float(message.content.split('temp ')[1])
                self.temp = max(0.1, temp_value)
                logger.info("NEW TEMPKA", self.temp)
            except ValueError:
                await message.channel.send("INOOO NO YOU WRONGS IT YOU WRONGS IT")
        if message.content.startswith('beam '):
            try:
                temp_value = float(message.content.split('beam ')[1])
                self.beam = temp_value
                logger.info("NEW BEAMKA", self.beam)
            except ValueError:
                await message.channel.send("INOOO NO YOU WRONGS IT YOU WRONGS IT")
    
    
    
    
    
    # upper lyaer
    def gen_1_t5(self, text: str):
        if not self.top_layer:
            return text
        logger.info("YAHJ YAH")
        message = {
            "type": "gen",
            "text": text,
            "model": "T5-mihm-gc",
            "config" : {
                "temperature" : self.temp,
                'max_new_tokens': 500,
                'num_beams': 2,
            },
            "from": "hiran"
        }
        response = self.pack_and_send(self.top_layer_socket, message)
        logger.info(response)
        return response
    
    def pipe(self, reply, pipe= 0):   
        start_time = time.perf_counter()
        if pipe == 1:
            return self.gen_1_t5(reply)
        if pipe == 2:
            return reply
        reply, tags = self.tag_patterns(reply)
        oky = self.untag_patterns(self.gen_1_t5(reply), tags)
        end_time = time.perf_counter()
        
        logger.info(f"pipe gen time {end_time - start_time:.6f} sekonmd")
        return oky
    
    
    
    
        # Interfaceka o
    def gen_0(self, text, dmessage, learn, reply, store):
        if not self.markov_layer:
            return
        dummy = False
        guild = None
        if dmessage.guild is not None:
            guild = dmessage.guild.id
        if store:
            dummy = {
                "guild": guild,
                "channel": dmessage.channel.id,
                "author": dmessage.author.id,
                "created_at": dmessage.created_at.timestamp(),
                "content": dmessage.content
            }
        message = {
            "type" : "gen",
            "text": text,
            "dmessage": dummy,
            "learn": learn,
            "reply": reply,
            "store": store
        }
        logger.info(message)
        start_time = time.perf_counter()
        response = self.pack_and_send(self.markov_socket, message)
        end_time = time.perf_counter()
        logger.info(response)
        logger.info(f"marlopv gen time {end_time - start_time:.6f} sekonmd")
    
        return response    

    #specially filtering here,,,,
    def specially_kanal_filter(self, message, reply):
        if str(message.channel.id) in ['1281722746614845515']:
            reply = reply.split(" ")[0]
        return reply
    
    # d breads and butters
    async def reply(self, message, filtered_content, _learn:bool, _reply:bool, _store:bool, prefix = '', rep = False):
        async with message.channel.typing():
            if str(message.channel.id) not in self.chat_context:
                self.chat_context[str(message.channel.id)] = Context(5)
            self.chat_context[str(message.channel.id)].append(filtered_content)
            sample = self.chat_context[str(message.channel.id)].to_string()
            reply = self.gen_0(sample, message, _learn, _reply, _store)
            logger.info(f"Input Texty: {filtered_content}, Context: {sample}, generate: {reply}")

            #replyka            
            if reply is not None:
                reply = self.pipe(reply, self.get_pipe(message.channel.id))
                logger.info(f"pipe outpu:::: {reply}")
                reply = self.specially_kanal_filter(message, reply)
                if rep:
                    await message.reply(reply)
                else:
                    await message.channel.send(reply)

    async def on_message(self, message: discord.Message):
        _learn = False
        _store = False
        _reply = True

        #anti webhooker
        if hasattr(message, 'webhook_id'):
            if message.webhook_id:
                webhook = await self.fetch_webhook(message.webhook_id)
                if webhook.name in SPOMQA_WEBHOOKER_nAME:
                    return
                if message.webhook_id in SPOMQA_WEBHOOKER_ID:
                    return

        #boortoop
        if str(message.author) == BOT_USERNAME:
            return
        # ANTI SPOMQINSON
        if str(message.author) in SPOMQA or str(message.channel.id) in SPOMQA_CHANNEL:
            return
        
        await self.processka_comandkionson(message)
        #specially userphone transaction
        if str(message.channel.id) == '695007649141620754':
            self.last_message_time = discord.utils.utcnow()
        
        filtered_content = MessageFilter.filter_content(message)
        if filtered_content == '':
            return

        # Learn from private messages
        if message.guild is None and LEARN_FROM_DIRECT_MESSAGE:
            _store = True
            _learn = True

        # Learn from all server messages
        if message.guild is not None and str(message.guild.id) not in ANTI_LERN_SERCER:
            if str(message.channel) not in LEARN_CHANNEL_EXCEPTIONS:
                _store = True
                _learn = True

        # Learn from User
        if str(message.author) == ALWAYS_LEARN_FROM_USER:
            _store = True
            _learn = True

        # real-time learning
        if _learn:
            self.gen_0(filtered_content, message, _learn, False, _store)
            _learn = False
            _store = False
            
        
        # Random Reply alzo d blocked list
        if message.guild is not None:
            #bloqd server
            if str(message.guild.id) in SPOMQA_SERCER:
                return
            await self.voice_commands(message)
            
            #speciallywhey need prefix ya  or reply ya          
            if str(message.channel.id) in SPECIALLY_CHANNEL:
                await self.reply(message, 
                                    filtered_content, _learn, _reply, _store,
                                    SPECIALLY_CHANNEL[str(message.channel.id)]['prefix'], 
                                    SPECIALLY_CHANNEL[str(message.channel.id)]['reply'])
                    
            #chekingsons if anti d blocked/muted channel
            if message.channel.id not in self.badka:
                try:
                    #Userphon
                    if message.author.id == 247283454440374274:
                        #await self.reply(message, filtered_content)
                        #pattern1 = r"(.+#\d+) <:userphone:650883846581125142> (.*)"
                        #match1 = re.search(pattern, filtered_content)
                        #pattern2 = r"(.+) <:userphone:\d{18}> (.*)"
                        #match2 = re.search(pattern, filtered_content)
                        #if match2 or match1:
                        #    logger.info("YAP")
                        logger.info(filtered_content)
                        tex = filtered_content.split("<:userphone:650883846581125142>")
                        logger.info(tex)
                        if len(tex) > 1:
                            tex = tex[1]
                        else:
                            tex = filtered_content
                        await self.reply(message, tex,  _learn, _reply, _store)
                    else:
                        #hiran his mention check
                        rep = False
                        for mention in message.mentions:
                            if str(mention) == '797884527510290433':
                                rep = True
                        await self.reply(message, filtered_content,  _learn, _reply, _store, rep=rep)
                except Exception as e:
                    #logger.info(str(e))
                    #logger.info("Forbidden" in str(e) or "forbidden" in str(e).lower())
                    if "Forbidden" in str(e):
                        self.badka.append(message.channel.id)
                        logger.info(self.badka)
            return

        # Reply to private messages
        if message.guild is None:
            await self.reply(message, filtered_content,  _learn, _reply, _store)

client = DiscordClient(intents=None)
client.run(DISCORD_TOKEN)
