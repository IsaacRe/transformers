from datetime import datetime

from transformers.pipelines import pipeline
from scs.handler import SyntaxValidityCheckHandler, JSONSchemaCheckFactory
from transformers.generation.output_validity import get_token_vocab
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('4bit/vicuna-13b-GPTQ-4bit-128g')

pipe = pipeline(model='gpt2')
#pipe.model.generation_config = gen_config

#s = pipe('Below is info on the last US presidents in JSON format: ', max_length=50, num_beams=1, enforce_json=True, allow_outer_list=False, allow_empty=False)
#s = pipe('Below is info on the last US presidents in JSON format: ', num_beams=1, enforce_one_of="George,Isaac")
json_schema = """[]{
    name: string,
    year_elected: number,
    age_elected: number,
    term_length?: number
}"""

handler = SyntaxValidityCheckHandler(
    get_token_vocab(tokenizer),
    JSONSchemaCheckFactory(schema=json_schema),
    num_workers=4,
)

json_prompt = 'List US presidents in JSON format'
json_schema = """[]{
    name: string,
    age: number,
    year: number
}"""

one_of_prompt = 'Who is the first US president?'
options = 'George Washington,Abraham Lincoln'

s = pipe(json_prompt, enforce_json_schema=json_schema)
s = pipe(one_of_prompt, enforce_one_of=options)

# json = '[{"name":"George Washington","year_elected":1789'
# tokenized = tokenizer.encode(json[0], add_special_tokens=False)
# tokenized[0] = tokenizer.convert_tokens_to_ids('[')

# for tok in tokenized:
#     print(tok, end='')
#     handler.update([tok])
#     handler.process_invalid_next_tokens()

# next_toks = list(handler.await_invalid_next_tokens())
# suppress = [(b, i, s) for b, i, s in next_toks if s]
# allow = [(b, i, s) for b, i, s in next_toks if not s]

json = '[{"year_elected":1746',',"age_elected":46,"name":"George Washington","term_length":2}]'
tokenized = tokenizer.encode(json[0], add_special_tokens=False)
tokenized[0] = tokenizer.convert_tokens_to_ids('[')
tokenized2 = tokenizer.encode(json[1], add_special_tokens=False)
tokenized2[0] = tokenizer.convert_tokens_to_ids('1')

start = datetime.utcnow()
for tok in tokenized:
    print(tok, end='')
    next_toks = list(handler.await_invalid_next_tokens())
    suppress = [(b, i, s) for b, i, s in next_toks if s]
    allow = [(b, i, s) for b, i, s in next_toks if not s]
    handler.update([tok])
    handler.process_invalid_next_tokens()
print((datetime.utcnow() - start).total_seconds())

for tok in tokenized2:
    print(tok, end='')
    next_toks = list(handler.await_invalid_next_tokens())
    suppress = [(b, i, s) for b, i, s in next_toks if s]
    allow = [(b, i, s) for b, i, s in next_toks if not s]
    handler.update([tok])
    handler.process_invalid_next_tokens()

start = datetime.utcnow()
s = pipe('Below is info on the last US presidents in JSON format: ', num_beams=1, max_new_tokens=10, enforce_json_schema=json_schema, constraint_workers=32)
print(s)
print((datetime.utcnow() - start).total_seconds())
# [{'generated_text': 'Who is the pope?\n\n{\n\n"Pope Benedict XVI" : 1}'}]

"""
Prompt:
Who is the pope?

Response:

{

"Francis" : "Pope John Paul II",

"Pauline Marriage " : "Pauline Marriage",

"Rosa" : "Rosa",

(max sequence length reached)"""