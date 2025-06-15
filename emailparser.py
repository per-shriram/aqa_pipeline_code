mport eml_parser
import json
from pathlib import Path

def extract_email_content(eml_file_path):
    """Extract structured content from EML file"""
    with open(eml_file_path, 'rb') as f:
        raw_email = f.read()
    
    ep = eml_parser.EmlParser()
    parsed_eml = ep.decode_email_bytes(raw_email)
    
    # Extract relevant fields
    email_data = {
        'sender': parsed_eml['header'].get('from', ''),
        'recipients': parsed_eml['header'].get('to', []),
        'subject': parsed_eml['header'].get('subject', ''),
        'date': parsed_eml['header'].get('date', ''),
        'message_id': parsed_eml['header'].get('message-id', ''),
        'body_text': '',
        'body_html': ''
    }
    
    # Extract body content
    for body in parsed_eml.get('body', []):
        if body.get('content_type') == 'text/plain':
            email_data['body_text'] = body.get('content', '')
        elif body.get('content_type') == 'text/html':
            email_data['body_html'] = body.get('content', '')
    
    return email_data
