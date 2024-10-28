import logging

# --- "User" Stuff Section ---
# ----------------------------

DISCORD_CLIENT_ID = 0
DISCORD_TOKEN = ""

BOT_USERNAME = ''

# Bot user ID
SELF_USER_ID = 0

# Learn from all servers and channels
LEARN_FROM_ALL = True

# Don't learn from any of these channels (by channel id)
LEARN_CHANNEL_EXCEPTIONS = []

# Learn from direct messages
LEARN_FROM_DIRECT_MESSAGE = True

# Always learn from a specific user no matter what other flags are set
# This should be set to a string containing a username like "SomeGuy#1234"
ALWAYS_LEARN_FROM_USER = None

# Randomly post messages in these servers (by server id)
RANDOM_MESSAGE_SERVER_ID = []

# Allows certain words to be posted in these servers, but not in servers that are not listed on this list
# (useful for privacy) (by server id)
PRIVATE_SERVER_ID = []

# Ignore the following users (use user strings, like 'Username#1111')
IGNORE_USERS = []

# Block the following from being posted and captured everywhere (using regular expressions)
BLOCK_PHRASE_ALL = (r'')

# Block the following to be posted on servers not in PRIVATE_SERVER_ID
BLOCK_PHRASE_PRIVACY = (r'()')

# ANTI LEARN SERCEr
ANTI_LERN_SERCER = []

#specially channel whey reply haves and suchka like a eg {'id': {"prefix": "prefix", "reply": True}}
SPECIALLY_CHANNEL = {}

#spqomsinons person anti (ALWOW), reply
DO_REMOVE_URL = False
SPOMQA = []
SPOMQA_CHANNEL = []
SPOMQA_SERCER = []

#specially channel whey userphone trigger ya 
USERPHONE_CHANNEL = 0

# ignore webhooker 
SPOMQA_WEBHOOKER_nAME = []
SPOMQA_WEBHOOKER_ID = []

#Tekob bad wordka
ANTI_ALWOW_WORD = []

# Removes phrases in BLOCK_PHRASE_ALL and BLOCK_PHRASE_PRIVACY
# WARNING! IT IS RECOMMENDED THAT YOU KEEP THIS AS IS FOR YOUR PRIVACY UNLESS YOU DON'T CARE
DO_REMOVE_PHRASE = False

# --- Technical Stuff Section ---
# -------------------------------
EARHOLER_PORT = "5557"
TOP_LAYER_PORT = "5556"
MARKOV_PORT = "5555"

GEN_TIME_LIMIT = 20000
HEART_BET_GREQ = 15

SEVER_LOG_LEVEL = logging.INFO