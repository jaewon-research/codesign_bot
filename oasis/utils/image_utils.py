# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""
Image utilities for multimodal support in OASIS.

This module provides utilities for handling images in social agent simulations,
including loading, encoding, and validating images for use with multimodal LLMs.
"""

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'
}

# Max image size (5MB by default for most LLM APIs)
MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


def detect_image_type(image_source: str) -> str:
    """
    Detect whether an image source is a file path, URL, or base64 encoded string.
    
    Args:
        image_source: The image source to detect
        
    Returns:
        One of: 'file', 'url', 'base64', or 'unknown'
    """
    if not image_source:
        return 'unknown'
    
    # Check if it's a URL
    parsed = urlparse(image_source)
    if parsed.scheme in ('http', 'https', 'ftp'):
        return 'url'
    
    # Check if it's a file path
    if os.path.exists(image_source):
        return 'file'
    
    # Check if it's base64 (starts with data: or looks like base64)
    if image_source.startswith('data:image'):
        return 'base64'
    
    # Try to decode as base64 to verify
    try:
        # Base64 should be ASCII and have proper padding
        if len(image_source) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in image_source):
            base64.b64decode(image_source)
            return 'base64'
    except Exception:
        pass
    
    return 'unknown'


def validate_image_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    # Check file extension
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        return False, f"Unsupported image format: {file_ext}. Supported formats: {SUPPORTED_IMAGE_FORMATS}"
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > MAX_IMAGE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, f"Image too large: {size_mb:.2f}MB. Maximum size: {MAX_IMAGE_SIZE_MB}MB"
    
    if file_size == 0:
        return False, "Image file is empty"
    
    return True, None


def get_image_mime_type(file_path: str) -> str:
    """
    Get the MIME type of an image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        MIME type string (e.g., 'image/jpeg')
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('image/'):
        return mime_type
    
    # Fallback based on extension
    ext = Path(file_path).suffix.lower()
    ext_to_mime = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    return ext_to_mime.get(ext, 'image/jpeg')


def encode_image_to_base64(file_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Base64 encoded string, or None if encoding failed
    """
    try:
        # Validate the image
        is_valid, error_msg = validate_image_file(file_path)
        if not is_valid:
            logger.error(f"Image validation failed: {error_msg}")
            return None
        
        # Read and encode the image
        with open(file_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        logger.info(f"Successfully encoded image: {file_path}")
        return encoded_string
        
    except Exception as e:
        logger.error(f"Failed to encode image {file_path}: {e}")
        return None


def create_image_content_block(image_source: str, image_type: Optional[str] = None) -> Optional[dict]:
    """
    Create an image content block suitable for multimodal LLM APIs.
    
    This function creates a standardized image content block that can be used
    with various LLM APIs (Claude, GPT-4V, etc.).
    
    Args:
        image_source: The image source (file path, URL, or base64 string)
        image_type: Optional type hint ('file', 'url', 'base64'). If None, will auto-detect.
        
    Returns:
        Dictionary with image content block, or None if processing failed
        
    Example return value:
        {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/jpeg',
                'data': '<base64-encoded-image>'
            }
        }
    """
    if not image_source:
        logger.warning("Empty image source provided")
        return None
    
    # Auto-detect image type if not provided
    if image_type is None:
        image_type = detect_image_type(image_source)
    
    try:
        if image_type == 'file':
            # Load and encode file
            base64_data = encode_image_to_base64(image_source)
            if not base64_data:
                return None
            
            mime_type = get_image_mime_type(image_source)
            return {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': mime_type,
                    'data': base64_data
                }
            }
            
        elif image_type == 'url':
            # URL-based image
            return {
                'type': 'image',
                'source': {
                    'type': 'url',
                    'url': image_source
                }
            }
            
        elif image_type == 'base64':
            # Already base64 encoded
            # Try to extract media type from data URI if present
            if image_source.startswith('data:'):
                # Format: data:image/jpeg;base64,<data>
                parts = image_source.split(',', 1)
                if len(parts) == 2:
                    media_type = parts[0].split(';')[0].replace('data:', '')
                    base64_data = parts[1]
                else:
                    media_type = 'image/jpeg'
                    base64_data = image_source
            else:
                media_type = 'image/jpeg'  # Default
                base64_data = image_source
            
            return {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': media_type,
                    'data': base64_data
                }
            }
        else:
            logger.error(f"Unknown image type: {image_type}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create image content block: {e}")
        return None


def prepare_multimodal_message(text_content: str, image_source: Optional[str] = None, 
                               image_type: Optional[str] = None) -> list:
    """
    Prepare a multimodal message with text and optional image content.
    
    Args:
        text_content: The text content of the message
        image_source: Optional image source (file path, URL, or base64)
        image_type: Optional type hint for the image
        
    Returns:
        List of content blocks suitable for multimodal LLM APIs
        
    Example:
        >>> prepare_multimodal_message("Describe this image", "path/to/image.jpg")
        [
            {'type': 'image', 'source': {...}},
            {'type': 'text', 'text': 'Describe this image'}
        ]
    """
    content_blocks = []
    
    # Add image first if provided
    if image_source:
        image_block = create_image_content_block(image_source, image_type)
        if image_block:
            content_blocks.append(image_block)
        else:
            logger.warning("Failed to create image content block, proceeding with text only")
    
    # Add text content
    content_blocks.append({
        'type': 'text',
        'text': text_content
    })
    
    return content_blocks

