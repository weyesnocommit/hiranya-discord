from config import ANTI_ALWOW_WORD
import discord
import re

ANTI_ALWOW = {}

for word in ANTI_ALWOW_WORD:
	regex = re.compile(re.escape(word), re.IGNORECASE)
	ANTI_ALWOW[regex] = 'rascal'
ANTI_ALWOW[re.compile(r'([🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿]{2,})')] = "dzhorzh"

class MessageFilter(object):
	@staticmethod
	def filter_content(message: str):
		filtered_content = message
		# Prevents words (declared under bot_config.py) from being captured into the bot's database
		#filtered_content = re.sub(BLOCK_PHRASE_ALL, '', filtered_content, flags=re.IGNORECASE)
		# Prevents groups of regional indicator emojis of size 2 or larger from being captured
		# (breaks the bot)
		#filtered_content = re.sub(r'([🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿]{2,})', '', filtered_content)
		# Prevents mentions and emails (plus some other things) from being captured
		#filtered_content = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|'
		#                         r'(<@[!&]?\d+>)|'
		#                        r'(<#\d+>)|'
		#                       r'(\bowoh\b)|(\bowob\b)', '', filtered_content)
		# Shortens groups of more than one whitespace character to a single whitespace character
		#filtered_content = re.sub(r' +', ' ', filtered_content)
		#filtered_content = filtered_content.strip()
    
		for bad_word_pattern, replacement in ANTI_ALWOW.items():
			filtered_content = bad_word_pattern.sub(replacement, filtered_content)
		return filtered_content
