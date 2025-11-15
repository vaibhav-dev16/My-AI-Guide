tools = [{
    "type": "function",
    "function": {
        "name": "make_notes",
        "description": "Create or update a document named 'important_document.docx' to store questions or information the LLM could not answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message or question that could not be answered, along with the reason or details."
                }
            },
            "required": ["message"]
        }
    }
}]
