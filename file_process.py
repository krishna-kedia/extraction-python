import fitz
import os
import base64


def encode_image(image_path: str) -> str:
    """
    Encodes an image file into a base64 string.

    This function reads an image from the specified file path and encodes its binary content 
    into a base64-encoded string. The result can be used for embedding images in JSON, HTML, 
    or sending over APIs.

    Args:
        image_path (str): The file path to the image to be encoded.

    Returns:
        str: A base64-encoded string representing the image.

    Raises:
        FileNotFoundError: If the specified image file does not exist.

    Example:
        ```python
        encoded_image = encode_image("documents/page1.jpg")
        print(encoded_image[:100])  # Print first 100 characters of the encoded string
        ```
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def to_base64(pdf_path: str) -> list:
    """
    Converts each page of a PDF file into a base64-encoded PNG image.

    This function reads the input PDF file, renders each page as a high-resolution
    image (300 DPI), and encodes each page as a base64 string in PNG format.

    Args:
        pdf_path (str): The file path to the PDF document to be processed.

    Returns:
        list: A list of base64-encoded strings, where each string represents
              a page of the PDF as a PNG image.
    Example:
        ```python
        base64_images = to_base64("documents/sample.pdf")
        print(base64_images[0][:100])  # Print first 100 characters of the first page's encoded string
        ```
    """
    doc = fitz.open(pdf_path)

    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        
        img_bytes = pix.tobytes("png")
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        images.append(base64_str)

    return images


def base_64_conversation(input_type: str = "PDF", file_path: str = "") -> list:
    """
    Converts a PDF or image file to base64-encoded string(s).

    For PDF files, each page is converted into a base64-encoded PNG image.
    For image files, the entire image is encoded as a single base64 string.

    Args:
        input_type (str): The type of input file. Accepted values are "PDF" and "IMAGE".
        file_path (str): The path to the file to be converted.

    Returns:
        list: A list of base64-encoded strings. For PDFs, each list item represents
              a page. For images, the list contains a single base64 string.

    Raises:
        ValueError: If the input type is not supported.
        FileNotFoundError: If the specified file does not exist.

    Example:
        ```python
        base64_images = base_64_conversation(input_type="PDF", file_path="documents/sample.pdf")
        print(base64_images[0][:100])  # Print first 100 characters of the first page's encoded string
        ```
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if input_type == "PDF":
        base64_pdf_pages = to_base64(file_path)
        return base64_pdf_pages
    elif input_type == "IMAGE":
        return [encode_image(file_path)]
    else:
        raise ValueError("Unsupported file type. Use 'PDF' or 'IMAGE'.")


if __name__ == "__main__":
    # Example usage
    # base64_images = base_64_conversation(input_type="PDF", file_path="/home/murari/Desktop/extract/ec_documents/152.pdf")
    # print(base64_images[0][:100]) 
    pass# Print first 100 characters of the first page's encoded string
