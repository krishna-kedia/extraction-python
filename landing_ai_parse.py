from agentic_doc.parse import parse_and_save_documents, parse_documents
from prompt_processing import load_json_schema
import asyncio
import json, os
from constants import VISION_AGENT_API_KEY

def retrieve_page_wise_parse(parsed_result: dict) -> list[str]:
    chunks = parsed_result["chunks"]
    page_level_separator = {}

    for chunk in chunks:
        page_number = chunk["grounding"][0]["page"]
        if page_number not in page_level_separator:
            page_level_separator[page_number] = [chunk]
        else:
            page_level_separator[page_number].append(chunk)
    
    page_level_text = []

    for page_number, page_chunks in page_level_separator.items():
        page_prompt = ""
        for chunk_in_page in page_chunks:
            chunk_type = chunk_in_page.get("chunk_type", "unknown")
            text = chunk_in_page.get("text", "").strip()
            page_prompt += f"<<{chunk_type}>>\n{text}\n\n"
        page_level_text.append(page_prompt)
    
    return page_level_text



async def landing_ai_vision_parser(document_path_or_url: list[str], save_parse: bool = False, result_save_dir: str = "") -> list[dict]:
    """
    Parses a document using the Landing AI Vision Parser.

    Args:
        document_path_or_url (list(str): List of paths or URLs of the document to be parsed.
        save_parse (bool): Boolean value indeicating to save the parsed JSONs or not
        result_save_dir (str): Directory path to save the results of the parsed JSONs.

    Returns:
        str: The path to the saved parsed result.
    """
    # Parse the document and save the result
    if not VISION_AGENT_API_KEY:
        raise ValueError("Missing VISION_AGENT_API_KEY. Please set it in a .env file.")
    
    if save_parse:
        parsed_result_path = parse_and_save_documents(documents=document_path_or_url, result_save_dir=result_save_dir)
        parsed_results = []

        for parses in parsed_result_path:
            parsed_results.append(load_json_schema(parsed_result_path))
    else:
        parsed_results_objects = parse_documents(documents=document_path_or_url)
        parsed_results = []
        
        for parses in parsed_results_objects:
            parsed_results.append(parses.model_dump())

    return parsed_results


if __name__ == "__main__":
    # Example usage
    async def main():
        # Path to the document to be parsed
        # document_path = ["/home/murari/Desktop/document-extraction-module/ec_documents/152.pdf", "/home/murari/Desktop/document-extraction-module/ec_documents/152_double.pdf"]
        # parsed_result = await landing_ai_vision_parser(document_path)
        
        # pdf_level_text = []
        # for pdf in parsed_result:
        #     page_level_text = retrieve_page_wise_parse(parsed_result=pdf)
        #     pdf_level_text.append(page_level_text)
        
        # print(pdf_level_text)
        # print(len(pdf_level_text[0]), len(pdf_level_text[1]))
        pass
    
    asyncio.run(main())
