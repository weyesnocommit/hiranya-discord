import traceback
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
import asyncio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAMPLING_STRATEGIES = ['softmax', 'top_p', 'top_k', 'greedy', 'random', 'top_p_k']
MODELKAS = []

logging.basicConfig(
    level=logging.INFO,
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

    def __init__(self, size):
        self.size = size
        self.chat_context = [None] * size

    def is_in_contextka(self, message):
        if self.count < 1:
            return False
        for msg in self.chat_context:
            if msg is not None:
                if msg[2] == message:
                    return True
        return False

    def append(self, id, name, message):
        self.last = message
        if not self.is_in_contextka(message):
            self.chat_context[self.index] = (id, name, message)
            self.index = (self.index + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def get(self):
        return [x for x in self.chat_context if x is not None]
    
    def to_string(self, sep=""):
        items = self.get()
        if self.count > 1:
            weights = [2 ** i for i in range(self.count)]
            items = [items[i] for i in random.choices(
                range(self.count), weights=weights, k=min(self.count, 3))]
            items.append(self.last)
            new_array = [b for b in items if len(b) <= 4*len(self.last)]
            return f"{sep} ".join(list(set([item[2].capitalize() for item in new_array])))
        else:
            return self.last
            
    def sample_n(self, n=10, sep=". ", names = True):
        items = self.get()
        weights = np.exp(np.linspace(0, 1, len(items)))  # Exponential growth
        weights /= weights.sum()  # Normalize weights to sum to 1
        # Sample n items based on weights without replacement
        selected_indices = np.random.choice(len(items), size=min(len(items), n), replace=False, p=weights)
        print("AM SELKECTIONS", selected_indices, weights)
        sampled_items = [items[i] for i in sorted(selected_indices)]
        if names:
            result = f"{sep}".join(f"{i[1]}: {i[2]}" for i in sampled_items)
            return result
        else:
            result = f"{sep}".join(f"{i[2]}" for i in sampled_items)
            return result

    def last_n(self, n=3, sep=". ", names= True, namesep=": ", ids=False, filter_out_ids=None, reverse=False):
        if not filter_out_ids:
            filter_out_ids = []
            
        items = self.get()
        if len(items) < 3:
            return self.last

        last_three = items[-n:] #items[-4:-1] if len(items) >= 4 else items[:-1]
        if reverse:
            last_three = last_three[::-1]
        if names and not ids:
            result = f"{sep}".join([f"{item[1]}{namesep}{item[2]}" for item in last_three if item[0] not in filter_out_ids])
            return result
        if names and ids:
            result = f"{sep}".join([f"{item[0]}</uid>{item[1]}{namesep}{item[2]}" for item in last_three if item[0] not in filter_out_ids])
            return result
        else:
            result = f"{sep}".join([item[2] for item in last_three if item[0] not in filter_out_ids])
            return result

class ZMQClient:
    def __init__(self, port, layer_name, heartbeat_interval=60):
        self.port = port
        self.layer_name = layer_name
        self.heartbeat_interval = heartbeat_interval
        self.context = zmq.Context()
        self.socket = self.create_zmq_socket()
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.heartbeat_task())

    def create_zmq_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, GEN_TIME_LIMIT)
        socket.connect(f"tcp://127.0.0.1:{self.port}")
        return socket

    def safe_send(self, message):
        try:
            packed_data = msgpack.packb(message)
            self.socket.send(packed_data)
            response = self.socket.recv()
            return msgpack.unpackb(response)
        except zmq.Again as e:
            logging.error(f"Timeout while waiting for a response: {e}")
            return None
        except zmq.ZMQError as e:
            logging.error(f"ZMQ Error: {e}, attempting to reconnect...")
            self.reconnect_socket()
            return None
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            return None

    def reconnect_socket(self):
        logging.error(f"Reconnecting {self.layer_name} socket")
        self.socket.close()
        self.socket = self.create_zmq_socket()

    async def heartbeat_task(self):
        print("am try am2")
        while True:
            print("am try am")
            await asyncio.sleep(self.heartbeat_interval)
            try:
                response = self.safe_send({"from": "hiran", "type": "ping"})
                if response:
                    logging.info(f"Heartbeat response from {self.layer_name}: {response}")
                else:
                    logging.error(f"No response from {self.layer_name} during heartbeat")
            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error(f"Failed heartbeat for {self.layer_name}")
                self.reconnect_socket()

        
class DiscordClient(discord.Client):
    _channel_tasks = {}
    muhharaq = "business on the business"
    badka = []
    chat_context = {}
    markov_topics = {}
    pipes = {}
    llm_cfg =  {
        "temperature" : 1.9,
        'max_new_tokens': 512,
        'num_beams': 6,
        'repetition_penalty': 1.01,
        'top_p': 0.9,
        'top_k': 100
    }
    markov_cfg = {
        'markov': {
            'strategy': SAMPLING_STRATEGIES[1],
            "temperature": 1.1,
            'top_k': 100,
            'top_p': 0.9
        },
        'struct': {
            'strategy': SAMPLING_STRATEGIES[1],
            "temperature": 0.8,
            'top_k': 100,
            'top_p': 0.9
        }
    }
    
    harraq_filter = MessageFilter()
    _ok_webhooker = WEBHOOKER_WHITELISTKA
    task = "grammar: "

    def __init__(self, intents):
        super().__init__(intents=intents)
        self.model = None
        self.ready = False
        self.timeout_duration = 60*60*1
        self.message_channel = None
        self.last_message_time = discord.utils.utcnow()
        self.timer_started = False
        self.layer_state = {"LLM": True, "MARKOV": True}
        self.use_llm = True
        print(self.llm_cfg)
        
        self.patterns = {
            r'<#\d+>': 'DISCORD_CHANNEL',
            r'<@\d+>': 'DISCORD_MENTION',
            r'https?://\S+': 'URL',
            r'<:\w+:\d+>': 'DISCORD_EMOJI'
        }
        self.zmq_context = {
            "MARKOV": zmq.Context(),
            "LLM": zmq.Context()
        }
        self.LLM = ZMQClient(TOP_LAYER_PORT, "LLM", heartbeat_interval=HEART_BET_GREQ)
        self.MARKOV = ZMQClient(MARKOV_PORT, "MARKOV", heartbeat_interval=HEART_BET_GREQ)

    async def on_thread_join(self, thread):
        await thread.join()
        await thread.send(f"wecloomes mysekf")
        
    async def on_ready(self):
        global MODELKAS
        MODELKAS = self.LLM.safe_send({"from": "hiran", "type": "get_models"})
        self.model = MODELKAS[0]
        print(MODELKAS)
        print(" ===========================================AM READINGSONSS")
        self.ready = True
        self.timer_task.start()
        
        if USERPHONE_CHANNEL:
            await self.call_userphone()
            await self.userphon(self.get_channel(USERPHONE_CHANNEL))
  
    ###################### START TOFI USERPHONE yayf ######################
    
    async def call_userphone(self):
        channel = self.get_channel(USERPHONE_CHANNEL)
        await channel.send("HUHHARABIN")
        commands = await channel.application_commands()
        for command in commands:
            if command.name == "userphone":
                await command.__call__(channel=channel)
                break 
    
    async def userphon(self, channel):
        # Update the last message time and reset the timer
        self.last_message_time = discord.utils.utcnow()
        
        if not self.timer_started:
            self.message_channel = channel
            self.timer_started = True
            self.timer_task.restart()

    @tasks.loop(seconds=60)
    async def timer_task(self):
        if self.last_message_time is None:
            return
        
        elapsed_time = (discord.utils.utcnow() - self.last_message_time).total_seconds()
        if elapsed_time >= self.timeout_duration:
            try:
                if self.message_channel and USERPHONE_CHANNEL:
                    await self.call_userphone()
                self.last_message_time = discord.utils.utcnow()  # Reset the timer
            except Exception as e:
                logger.error(str(e))

    @timer_task.before_loop
    async def before_timer_task(self):
        await self.wait_until_ready()
    
    ###################### STOP TOFI USERPHONE yayf ######################
    
    ###################### START TOFI ZMQ stufyf ######################



        
        
        
        
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
        if not self.use_llm:
            return hubinta
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
        # hooker whitelistka
        if message.content.startswith('hooker '):
            try:
                hookid = int(message.content.split('hooker ')[1])
                self._ok_webhooker.append(hookid)
                logger.info("NEW HOOKER", self._ok_webhooker)
            except ValueError:
                await message.channel.send("NOTTT HOOKER IT")
                
       #antti anti opt outka
        if message.content.startswith('tarrachka tarrachka anti say me my laxanka number'):
            CHAEHCCHAECH = message.author.display_name
            cursor.execute("INSERT INTO users (user_id, username, nickname) VALUES (?, ?, ?)",
                        (message.author.id, message.author.name, CHAEHCCHAECH))
            conn.commit()
            msg = "REMOVKINSONS AM REMOVE YOU AM REMOVKINSONS AM REMOVED YOUSKA AM REMOVED YOU OKAY AM REMOVED YOUR AM NOT AM NOT AM YES IYESIYESIYES I YES DIDKINSON MR NEVISHI AM YES YES So, regarding your silly ping...  Did you consider that I'm a grown-ass man, with a life?   That perhaps your joke wouldn't be very funny to me? Granted, not much of a life, but I had to pause my youtube video, switch to discord, read back several pages to see what the deal was."
            reply = self.gen_0(msg, message, False, True, False)
            await message.channel.send(f"IYES ~~~~~ ~ ~ {reply}")

        #opt outka (antI)
        if message.content.startswith('tarrachka tarrachka say me my laxanka number'):
            cursor.execute("DELETE FROM users WHERE user_id=?", (message.author.id,))
            conn.commit()
            msg = "i will pings you mr  anushi i will pings i will pingkinsosn like a pinkie finger"
            reply = self.gen_0(msg, message, False, True, False)
            await message.channel.send(f"I MOT !!!!11 ```cool 111```{reply}")
            
        # tekob anti reply gen channelka
        if 'clearkalistka' in message.content:
            self.badka = []
            
        #specially pipeline
        pipe_numbet = self.pipeka_numberka(message.content)
        if pipe_numbet:
            self.pipes[message.channel.id] = pipe_numbet

        # markovcfg
        if message.content.startswith('markov_temp '):
            try:
                temp_value = float(message.content.split('markov_temp ')[1])
                self.markov_cfg['markov']['temperature'] = max(0.1, temp_value)
                logger.info(f"NEW MARKOV CFG {self.markov_cfg}")
            except Exception as e:
                logger.info(e)
                await message.channel.send("AM NITTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH")

        if message.content.startswith('markov_strategy '):
            try:
                temp_value = message.content.split('markov_strategy ')[1]
                if temp_value in SAMPLING_STRATEGIES:
                    self.markov_cfg['markov']['strategy'] = temp_value
                    logger.info(f"NEW MARKOV CFG {self.markov_cfg}")
            except Exception as e:
                logger.info(e)
                await message.channel.send("AM NITTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH")

        if message.content.startswith('markov_topp '):
            try:
                temp_value = float(message.content.split('markov_topp ')[1])
                self.markov_cfg['markov']['top_p'] = min(max(0.1, temp_value), 1.0)
                logger.info(f"NEW MARKOV CFG {self.markov_cfg}")
            except Exception as e:
                logger.info(e)
                await message.channel.send("AM NITTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH")

        if message.content.startswith('markov_topk '):
            try:
                temp_value = int(message.content.split('markov_topk ')[1])
                self.markov_cfg['markov']['top_k'] = max(1, temp_value)
                logger.info(f"NEW MARKOV CFG {self.markov_cfg}")
            except Exception as e:
                logger.info(e)
                await message

        if message.content.startswith('struct_temp '):
            try:
                temp_value = float(message.content.split('struct_temp ')[1])
                self.markov_cfg['struct']['temperature'] = max(0.1, temp_value)
                logger.info(f"NEW MARKOV CFG {self.markov_cfg}")
            except Exception as e:
                logger.info(e)
                await message.channel.send("AM NITTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH")

        if message.content == 'llmkaoffka':
            self.use_llm = False
        if message.content == 'llmkaonka':
            self.use_llm = True


        # llmcfg
        if message.content.startswith('temp '):
            try:
                temp_value = float(message.content.split('temp ')[1])
                self.llm_cfg['temperature'] = max(0.1, temp_value)
                logger.info(f"NEW TEMPKA {self.llm_cfg['temperature']}")
            except ValueError:
                await message.channel.send("INOOO NO YOU WRONGS IT YOU WRONGS IT")

        #beamka
        if message.content.startswith('beam '):
            try:
                temp_value = int(message.content.split('beam ')[1])
                self.llm_cfg['num_beams'] = temp_value
                logger.info("NEW BEAMKA", self.beam)
            except ValueError:
                await message.channel.send("INOOO NO YOU WRONGS IT YOU WRONGS IT")
    
        if message.content.startswith('modelka '):
            try:
                model = message.content.split("modelka ")[1]
                if model in MODELKAS:
                    self.model = model
            except:
                await message.channel.send("YOU NOT CORECT")
        
        if message.content.startswith('task '):
            try:
                task = message.content.split("task ")[1]
                if task in ['grammar', 'complete', 'translate English to German', 'translate English to Chinese']:
                    self.task = task + ": "
                if task == "none":
                    self.task = ""
                logger.info(f"NEW TASKSA {self.task}")
                
            except:
                await message.channel.send("NOT WRONGS ITS YOU")
    
    def repair_discord_mentions(self, text):
        emoji_pattern = r"(?<!<)[-:](\w+)[-:](\d+)(>?)"
        mention_pattern = r"(?<!<)[@#&\-](\w+|\d+)[\-:](\d+)(>?)"
        repaired_text = re.sub(emoji_pattern, r"<:\1:\2>", text)

        def mention_replacer(match):
            prefix = match.group(0)[0]  # Extract the type symbol (:, @, #, or &)
            if prefix == ':':  # Emoji
                return f"<:{match.group(1)}:{match.group(2)}>"
            elif prefix == '@':  # User or role
                if '&' in match.group(0):  # Role mention
                    return f"<@&{match.group(2)}>"
                return f"<@{match.group(2)}>"  # User mention
            elif prefix == '#':  # Channel mention
                return f"<#{match.group(2)}>"
            return match.group(0)

        repaired_text = re.sub(mention_pattern, mention_replacer, repaired_text)
        repaired_text = re.sub(r"(>){2,}", ">", repaired_text)
        return repaired_text
    
    
    def fix_refs(self, message):
        broken_mention_pattern = r'(?<!<)([@#&]\d+>)'
        fixed_message = re.sub(broken_mention_pattern, r'<\1', message)
        return fixed_message
    
    def input_fromatter(self, text, context = "", author="harraq", author_id=138308895834767360):
        if not self.layer_state['LLM']:
            return text
        formatted = text
        ctx = ""
        if self.model == 't5-mihm':
            return self.task+text
        elif self.model == 't5-gcc-03':
            text = text.replace("grammar", "NOTT")
            ctx = context.last_n(7, sep=chr(10))# context.replace("grammar", 'nott')
            formatted = f'grammar: {text} | {ctx}'
        elif self.model == 'flant5-gcc-01':
            ctx = context.last_n(15, sep=chr(10))
            formatted = f'grammar: {text} context:\n\n{ctx} </s>'
        elif self.model in ['t5-cg-1.0',  "checkpoint-16500"]:
            ctx = context.last_n(7, sep=chr(10), names=True, namesep="</msg>", ids=True)
            formatted = f'grammar: {text} context:\n\n{ctx} </s>'
        elif self.model == 't5-common-gen':
            ctx = context.last_n(3, sep=" ")
            formatted = f'{text}'
        elif self.model == 't5-cbcg':
            ctx = context.last_n(7, sep=chr(10), names=True, namesep="</msg>", ids=True, reverse=True)
            formatted = f'grammar:</uid>{author_id}</name>{author}</msg>{text.replace("<", "&lt;")}</end>{ctx} </s>'
        return formatted, ctx
        
    def output_fromatter(self, text):
        if not self.layer_state['LLM']:
            return text
        if self.model == 't5-mihm':
            pattern = r"(?<!<):(\w+):(\d+)(?=>|$)"
            repaired_text = re.sub(pattern, r"<:\1:\2>", text)
            return repaired_text
        elif self.model in ['t5-gcc-03', 't5-cg-1.0']:
            return self.fix_refs(self.repair_discord_mentions(text))
        elif self.model == 't5-cbcg':
            return text.replace("&lt;", "<")
        else:
            return text
        
    # upper lyaer
    def gen_1_t5(self, text: str):
        print("AM HERES", self.layer_state)
        if not self.layer_state['LLM']:
            return text
        if not text:
            return None
        message = {
            "type": "gen",
            "text": text,
            "model": self.model,
            "config" : self.llm_cfg,
            "from": "hiran"
        }
        response = self.LLM.safe_send(message)
        logger.info(response)
        return response
    
    def pipe(self, input, pipe= 0):   
        start_time = time.perf_counter()
        
        #if pipe == 1:
        #    return self.gen_1_t5(input)
        #if pipe == 2:
        #    return reply
        #reply, tags = self.tag_patterns(input)
        #oky = self.untag_patterns(self.gen_1_t5(input), tags)
        oky = self.gen_1_t5(input)
        end_time = time.perf_counter()
        
        logger.info(f"pipe gen time {end_time - start_time:.6f} sekonmd")
        return oky
    
    
    
    
        # Interfaceka o
    def gen_0(self, text, dmessage, learn, reply, store):
        if not self.layer_state['MARKOV'] or not text:
            logger.info(f"is availables MARKOVKA: {self.layer_state['MARKOV']}")
            return
        dummy = False
        guild = None
        if store:
            if dmessage.guild is not None:
                guild = dmessage.guild.id
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
            "store": store,
            "sampling_config": self.markov_cfg
        }
        logger.info(message)
        start_time = time.perf_counter()
        response = self.MARKOV.safe_send(message)
        end_time = time.perf_counter()
        logger.info(response)
        logger.info(f"marlopv gen time {end_time - start_time:.6f} sekonmd")
    
        return response    

    #specially filtering here,,,,
    def specially_kanal_filter(self, message, reply):
        if str(message.channel.id) in ['1281722746614845515']:
            reply = reply.split(" ")[0]
        return reply
    
    def get_name(self, message):
        try:
            n = message.author.nick
            if n:
                return n
            else:
                return message.author.name
        except:
            return message.author.name
    
    def check_similarity(self, message, reply_from_pipe):
        """
        Compare the similarity between the input message and the reply from the pipe.
        """
        vectorizer = TfidfVectorizer().fit_transform([message, reply_from_pipe])
        similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        return similarity_matrix[0][0]
    
    def get_author(self, message):
        author = message.author.name
        try:
            author_ = message.author.nick
            if author_ is not None:
                author = author_
            else:
                author = message.author.display_name
        except:
            1+1
        return author

    # d breads and butters
    async def reply(self, message, filtered_content, _learn: bool, _reply: bool, _store: bool, prefix='', rep=False):
        async with message.channel.typing():
            channel_id = str(message.channel.id)
            sample = ". ".join(self.markov_topics[channel_id])
            self.markov_topics[channel_id] = []
            
            reply = self.gen_0(sample, message, _learn, _reply, _store)
            if reply is not None:
                logger.info(f"\nINPUT: {filtered_content}\nCONTX: {sample}\nMARKO: {reply}")
                ctx = self.chat_context[channel_id]
                ctx_authorka = message.author.name
                ctx_authorka_id = message.author.id
                input, cnt_ = self.input_fromatter(text=reply, context=ctx, author=ctx_authorka, author_id=ctx_authorka_id)
                
                reply_from_pipe = self.pipe(input, self.get_pipe(message.channel.id))
                logger.info(f"\nCONTX: {cnt_}\nLMGEN: {reply_from_pipe}")
                
                if reply_from_pipe is not None:
                    similarity = self.check_similarity(filtered_content, reply_from_pipe)
                    print(f"similarity: {filtered_content}, {reply_from_pipe}, {similarity}")
                    if similarity < 0.8:
                        reply = reply_from_pipe
                        reply = self.output_fromatter(reply)
                        
                    reply = self.harraq_filter.filter_content(reply)
                    if SELF_CONTEXT:
                        self.chat_context[channel_id].append(ctx_authorka_id, self.get_author(message), reply)
                    logger.info(f"pipe output:::: {reply}")
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
        if message.webhook_id and message.webhook_id not in self._ok_webhooker:
            if not message.webhook_id in WEBHOOKER_WHITELISTKA:
                return
            
        #boortoop
        if message.author.id == SELF_USER_ID:
            return
        
        # ANTI SPOMQINSON
        if str(message.author) in SPOMQA or str(message.channel.id) in SPOMQA_CHANNEL:
            return
        
        if message.guild is not None and str(message.guild.id) in SPOMQA_SERCER:
            return
        
        await self.processka_comandkionson(message)
        #specially userphone transaction
        if str(message.channel.id) == '695007649141620754':
            self.last_message_time = discord.utils.utcnow()
        
        filtered_content = self.harraq_filter.filter_content(message.content)
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
        
        channel_id = str(message.channel.id)
                
        if channel_id not in self.chat_context:
            self.chat_context[channel_id] = Context(150)
        self.chat_context[channel_id].append(message.author.id, self.get_name(message), filtered_content)
        
        if channel_id not in self.markov_topics:
            self.markov_topics[channel_id] = []
        self.markov_topics[channel_id].append(filtered_content)

        if channel_id in self._channel_tasks and not self._channel_tasks[channel_id].done():
            print(f"Task for channel {channel_id} is busy")
            return

        self._channel_tasks[channel_id] = asyncio.create_task(
            self._on_message(message, filtered_content, _learn, _store, _reply)
        )
        
    async def _on_message(self, message: discord.Message, filtered_content, _learn, _store, _reply):
       
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
                        tex = filtered_content.split("<:userphone:1311268018625576971>")
                        logger.info(tex)
                        if len(tex) > 1:
                            tex = tex[1]
                        else:
                            tex = filtered_content
                        logger.info(f"USERPHON {tex}")
                        message.content = tex
                        #await self.reply(message, tex,  _learn, _reply, _store)
                        #hiran his mention check
                    rep = False
                    for mention in message.mentions:
                        if str(mention) == '797884527510290433':
                            rep = True
                    await self.reply(message, filtered_content,  _learn, _reply, _store, rep=rep)
                except Exception as e:
                    logger.error(str(e))
                    logger.error(traceback.format_exc())
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
