from io import StringIO

import sampling
import rwkv_cpp_model
import rwkv_cpp_shared_library
from rwkv_tokenizer import get_tokenizer

# ======================================== Script settings ========================================

# English, Chinese, Japanese
LANGUAGE: str = 'Chinese'
# QA: Question and Answer prompt to talk to an AI assistant.
# Chat: chat prompt (need a large model for adequate quality, 7B+).
PROMPT_TYPE: str = 'Chat'

MAX_GENERATION_LENGTH: int = 250

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 0.8
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.5
# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 0.2
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
FREQUENCY_PENALTY: float = 0.2

END_OF_LINE_TOKEN: int = 187
DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0

# =================================================================================================

tokenizer_decode, tokenizer_encode = get_tokenizer('20B')
user, bot, separator = 'Bob', 'Alice', ':'

model = None
model_path = 'D:\\llm_store\\rwkv\\RWKV-4b-Pile-171M-20230202-7922-Q8_0.bin'
model_path = 'D:\\llm_store\\rwkv\\RWKV-4-Raven-7B-v12-20230530-ctx8192-Q8_0.bin'

def get_model():
    global model, model_path

    if model is None:
        library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        model = rwkv_cpp_model.RWKVModel(library, model_path)

    return model

def split_last_end_of_line(tokens):
    if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
        tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]

    return tokens

class Chat:
    def __init__(self, session_id, init_prompts):
        self._session_id = session_id
        self._init_prompts = init_prompts

        self._model = get_model()
        self._inited = False
    
        self._processed_tokens = []
        self._logits = None
        self._state = None

        self._init_session()
    
    def _process_tokens(self, tokens, new_line_logit_bias = 0.0):
        self._processed_tokens += tokens

        for token in tokens:
            self._logits, self._state = self._model.eval(token, self._state, self._state, self._logits)
        
        self._logits[END_OF_LINE_TOKEN] += new_line_logit_bias
    
    def _init_session(self):
        if self._inited:
            return

        tokens = tokenizer_encode(self._init_prompts)
        self._process_tokens(split_last_end_of_line(tokens))
        self._inited = True

    
    def _parse_response(self):
        temperature = TEMPERATURE
        top_p = TOP_P

        start_index = len(self._processed_tokens)
        accumulated_tokens = []
        token_counts = {}

        for i in range(MAX_GENERATION_LENGTH):
            for n in token_counts:
                self._logits[n] -= PRESENCE_PENALTY + token_counts[n] + FREQUENCY_PENALTY
            
            token = sampling.sample_logits(self._logits, temperature, top_p)

            if token == END_OF_LINE_TOKEN:
                break
            
            if token not in token_counts:
                token_counts[token] = 1
            else:
                token_counts[token] = +1
            
            self._process_tokens([token])
            accumulated_tokens += [token]

            decoded = tokenizer_decode(accumulated_tokens)

            if '\uFFFD' not in decoded:
                accumulated_tokens = []
                yield decoded
            
            if '\n\n' in tokenizer_decode(self._processed_tokens[start_index:]):
                break
            
            if i == MAX_GENERATION_LENGTH - 1:
                break
    
    def eval(self, msg):
        global user, bot, separator

        rendered = f'{user}{separator} {msg}\n\n{bot}{separator}'
        self._process_tokens(tokenizer_encode(rendered), new_line_logit_bias=-999999999)

        return self._parse_response()

session_store = {}

def get_init_prompt():
    global user, bot

    return f"""
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
{bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. \
{bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.\n
"""

def get_session(id):
    global session_store

    session = session_store.get(id)
    if session is None:
        session_store[id] = Chat(id, get_init_prompt())
    
    return session_store.get(id)