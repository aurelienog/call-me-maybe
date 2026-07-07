*This project has been created as part of the 42 curriculum by aunoguei.*

# Call Me Maybe

## Description

## Instructions

## Algorithm explanation:

Describe your constrained decoding approach in detail

## Design decisions:

Explain key choices in your implementation

## Performance analysis:

Discuss accuracy, speed, and reliability of your solution

## Challenges faced:

Document difficulties encountered and how you solved them

## Testing strategy:

Describe how you validated your implementation

## Example usage:

Provide clear examples of running your program

## Resources

UV:
- https://docs.astral.sh/uv/
- https://www.datacamp.com/tutorial/python-uv

Pydantic:
- https://pydantic.dev/docs/

JSON data:
- https://www.youtube.com/watch?v=9N6a-VLBa2I

LLM:
- https://www.youtube.com/watch?v=kCc8FmEb1nY



https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/

https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20

https://connorshorten300.medium.com/structured-outputs-the-building-blocks-of-reliable-ai-1d7e56f99a94


La regla que sigo siempre es muy sencilla:

Si una clase representa datos (algo que viene de un JSON o que vas a serializar), uso Pydantic.
Si una clase representa comportamiento o lógica (un registro, un algoritmo, un decoder), uso una clase normal.

Creo que esa separación hace el diseño mucho más limpio y encaja muy bien con este proyecto.