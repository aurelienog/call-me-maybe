texto
    ↓
tokenizer
    ↓
modelo
    ↓
logits

El SDK devuelve:

logits = model.get_logits_from_input_ids(...)

V.3.1 The LLM SDK
Attached to this project, you’ll find a wrapper class Small_LLM_Model in the llm_sdk
package that you can use to interact with the LLM.
The SDK provides several essential methods:
• get_logits_from_input_ids(input_ids: List[int]) -> List[float]
Takes a list of token IDs and returns the logits produced by the LLM model.
• get_path_to_vocab_file() -> str
Returns the path to the vocabulary file containing the correspondence between
token IDs and tokens.
• encode(text: str) -> Tensor
Encodes a text string into a tensor of token IDs using the model’s tokenizer.
• decode(token_ids: List[int]) -> str (optional)
Optionally decodes a list of token IDs back into a text string