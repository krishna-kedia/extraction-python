"""
prompt_processing.py
This module defines the functions which are used to generate prompts from the classes. 
This prompt is used to extract the data from the document.
The `PromptSchema` class is a Pydantic model that defines the structure of the prompt schema.
It includes fields for AI agent information, field definitions to extract, and an example output which is optional.
"""
from pydantic import BaseModel, Field
from typing import get_args, get_origin, List, Optional, Dict, Any
import json

class PromptSchema(BaseModel):
    """PromptSchema defines the current structure of the prompt schema which is used to extract the data from the document."""
    ai_agent_information: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Meta info or guidance for the AI agent (e.g., task, scope, constraints)."
    )
    extract_fields: Dict[Any, Any] = Field(
        ...,
        description="Field definitions with types and descriptions to extract."
    )
    output_example: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Example output data in the expected format."
    )


def load_json_schema(file_path: str) -> Dict[Any, Any]:
    """Load a JSON schema from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema_dict = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")    
    return schema_dict


def generate_prompt_from_schema(schema: PromptSchema) -> str:
    """
    Generates a structured natural language prompt string from a schema definition.

    This function composes a complete prompt based on AI agent instructions, field
    extraction schema, and example outputs. The resulting string is suitable for 
    input to a language model for information extraction tasks.

    Args:
        schema (PromptSchema): An instance of the `PromptSchema` Pydantic model
            containing the following optional components:
            - `ai_agent_information`: A dictionary of instructional text or notes.
            - `extract_fields`: A dictionary defining the field structure to be extracted.
            - `output_example`: A dictionary showing a sample expected JSON output.
        All these fields are defined as Pydantic classes in extraction_class_types.py.

    Returns:
        str: A well-formatted string prompt combining agent instructions, field
            schema, and an output example, intended for use with LLM-based extraction.

    Raises:
        TypeError: If the schema input is not an instance of `PromptSchema`.

    Example:
        >>> prompt = generate_prompt_from_schema(my_schema)
        >>> print(prompt)
        AI Agent Instructions:
        ...

        🔍 Fields to Extract in the JSON format as shown below:
        {
          "field1": "<str>",
          ...
        }

        🧾 Output Example:
        ```json
        {
          "field1": "value",
          ...
        }
        ```
    """
    prompt_parts = []

    # Add base information, instruction, and notes
    if "ai_agent_information" in schema and schema["ai_agent_information"]:
        for key, value in schema["ai_agent_information"].items():
            prompt_parts.append(f"{key}:\n{value['description']}\n")

    # Extract field definitions
    if "extract_fields" in schema and schema["extract_fields"]:
        prompt_parts.append("🔍 Fields to Extract in the JSON format as shown below:")
        prompt_parts.append("".join(f"{json.dumps(schema['extract_fields'], indent=2)}\n"))

    # Add output example
    if "output_example" in schema and schema["output_example"]:
        prompt_parts.append(f"\n🧾 Output Example:\n```json\n{json.dumps(schema['output_example'], indent=4, ensure_ascii=False)}\n```")

    return "\n\n".join(prompt_parts)


def generate_prompt_template(model: BaseModel) -> dict:
    """
    Generates a structured schema-like dictionary from a Pydantic model for use in prompt templates.

    This function recursively parses a Pydantic model and builds a nested dictionary
    representing each field's type and description. It supports nested models, lists,
    and primitive data types, making it useful for constructing LLM-friendly prompts or
    documentation schemas.

    Args:
        model (BaseModel): A Pydantic model class that defines the schema for which a
            prompt template should be generated.

    Returns:
        dict: A nested dictionary structure where each key corresponds to a field name
        and maps to a dictionary containing the field's `type` and `description`. 
        Nested models and lists are recursively expanded into their respective structures.

    Example:
        >>> class Person(BaseModel):
        ...     name: str = Field(..., description="The person's full name.")
        ...     age: int = Field(..., description="The person's age.")
        >>> generate_prompt_template(Person)
        {
            'name': {'type': 'str', 'description': "The person's full name."},
            'age': {'type': 'int', 'description': "The person's age."}
        }

    Notes:
        - Uses `model.model_fields` (Pydantic v2+) to iterate over fields.
        - Handles nested Pydantic models, lists of models, and primitive field types.
        - Requires the `Field(..., description="...")` format for field descriptions to be included.

    Raises:
        AttributeError: If the model fields cannot be accessed or do not conform to expected structure.
    """
    def recurse(fields):
        template = {}
        for name, field in fields.items():
            annotation = field.annotation
            origin = get_origin(annotation)
            description = field.description or ""

            # Handle list of nested models or primitives
            if origin in (list, List):
                item_type = get_args(annotation)[0]
                if hasattr(item_type, "model_fields"):
                    template[name] = {
                        "type": "array",
                        "description": description,
                        "items": recurse(item_type.model_fields)
                    }
                else:
                    template[name] = {
                        "type": f"array<{item_type.__name__}>",
                        "description": description
                    }

            # Handle nested model
            elif hasattr(annotation, "model_fields"):
                template[name] = {
                    "type": "object",
                    "description": description,
                    "properties": recurse(annotation.model_fields)
                }

            # Handle primitive fields
            else:
                try:
                    type_name = annotation.__name__
                except AttributeError:
                    type_name = str(annotation)
                template[name] = {
                    "type": type_name,
                    "description": description
                }
        return template

    return recurse(model.model_fields)


if __name__ == "__main__":
    
    class ExtractionClass(BaseModel):
        name: str = Field(..., description="The name of the person.")
        age: int = Field(..., description="The age of the person.")
        hobbies: List[str] = Field(..., description="A list of hobbies.")
        address: dict = Field(..., description="The address of the person.")

    class AIAgentClass(BaseModel):
        information: str = Field(..., description="Information for the AI agent.")
        instruction: str = Field(..., description="Instructions for the AI agent.")
        condition: str = Field(..., description="Conditions for the AI agent.")
    
    class OutputExampleClass(BaseModel):
        """"""
    
    extraction_template = generate_prompt_template(ExtractionClass)
    ai_agent_template = generate_prompt_template(AIAgentClass)
    output_example_class = generate_prompt_template(OutputExampleClass)
    
    prompt_schema = PromptSchema(
        ai_agent_information=ai_agent_template,
        extract_fields=extraction_template,
        output_example=output_example_class
    )

    prompt = generate_prompt_from_schema(prompt_schema.model_dump())
    print(prompt)
