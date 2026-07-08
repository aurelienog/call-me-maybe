from .models import Prompt, FunctionRegistry, FunctionCallResult


class ConstrainedDecoder():

    def process(self, prompts: Prompt, registry: FunctionRegistry) -> list[FunctionCallResult]:
        results = []
        for prompt in prompts:
            result = self.decode(prompt, registry)
            results.append(result)

    def decode(self, prompt: Prompt, registry) -> FunctionCallResult:
        return

allowed_tokens()

mask_logits()

select_next_token()

generate()

tokenize()

#     id_to_token = {
#     0: "<pad>",
#     1: "{",
#     ...
# }

# token_to_id = {
#     "{": 1,
#     ...
# }



                    Prompt
                       │
                       ▼
                 SmallLLMModel
                       │
          logits (numpy.ndarray)
                       │
                       ▼
             ConstrainedDecoder
                       │
        modifica el ndarray
                       │
                       ▼
            siguiente token válido