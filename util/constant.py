from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


ENCODE_TOKEN = '<ENCODE>' # Special token for encoder to generate embeddings
SOFT_PROMPT_START = '<SOFT_PROMPT_START>' # Special token for soft prompt start
SOFT_PROMPT_TOKEN = '<SOFT_PROMPT>' # Special token for soft prompt
SOFT_PROMPT_END= '<SOFT_PROMPT_END>' # Special token for soft prompt end

# Special token for future extension
RANK_TOKEN = '<RANK>'
RANK_START = '<RANK_START>'
RANK_END = '<RANK_END>'