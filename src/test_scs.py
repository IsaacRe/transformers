from datetime import datetime

from transformers.pipelines import pipeline
from scs.handler import SyntaxValidityCheckHandler, JSONSchemaCheckFactory
from transformers.generation.output_validity import get_token_vocab


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
    get_token_vocab(pipe.tokenizer),
    JSONSchemaCheckFactory(schema=json_schema),
    num_workers=4,
)

json = '[{"name":"George Washington","year_elected":1746,"age_elected":46,"term_length":2}]'
tokenized = pipe.tokenizer.encode(json)

start = datetime.utcnow()
for tok in tokenized:
    print(tok, end='')
    list(handler.await_invalid_next_tokens())
    handler.update([tok])
    handler.process_invalid_next_tokens()
print((datetime.utcnow() - start).total_seconds())

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