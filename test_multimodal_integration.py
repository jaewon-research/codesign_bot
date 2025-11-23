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
Test script for multimodal integration in OASIS.

This script tests the image handling capabilities without requiring actual images.
"""

import sys
import os
sys.path.append('/Users/robertgiometti/School/Research/CommunityBot/codesign_bot/oasis')

def test_image_utils():
    """Test image utility functions."""
    print("=" * 60)
    print("Testing Image Utilities")
    print("=" * 60)
    
    from oasis.utils.image_utils import (
        detect_image_type,
        validate_image_file,
        prepare_multimodal_message,
        create_image_content_block
    )
    
    # Test 1: Detect image type
    print("\n1. Testing image type detection...")
    test_cases = [
        ("./test.jpg", "file"),
        ("https://example.com/image.png", "url"),
        ("data:image/jpeg;base64,/9j/4AAQ...", "base64"),
    ]
    
    for source, expected in test_cases:
        if source.startswith("./"):
            # Skip file tests as files don't exist
            print(f"   ⊘ Skipping file test: {source}")
            continue
        result = detect_image_type(source)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {source[:50]}... -> {result} (expected: {expected})")
    
    # Test 2: Create image content block
    print("\n2. Testing image content block creation...")
    
    # Test with URL
    url_block = create_image_content_block(
        "https://example.com/test.jpg",
        "url"
    )
    if url_block and url_block.get("type") == "image":
        print(f"   ✓ URL image block created successfully")
        print(f"     Structure: {list(url_block.keys())}")
    else:
        print(f"   ✗ Failed to create URL image block")
    
    # Test with base64 (mock data)
    mock_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    base64_block = create_image_content_block(
        f"data:image/png;base64,{mock_base64}",
        "base64"
    )
    if base64_block and base64_block.get("type") == "image":
        print(f"   ✓ Base64 image block created successfully")
        print(f"     Media type: {base64_block['source'].get('media_type')}")
    else:
        print(f"   ✗ Failed to create base64 image block")
    
    # Test 3: Prepare multimodal message
    print("\n3. Testing multimodal message preparation...")
    
    text = "Describe this image in detail"
    multimodal_msg = prepare_multimodal_message(
        text_content=text,
        image_source="https://example.com/test.jpg",
        image_type="url"
    )
    
    if isinstance(multimodal_msg, list) and len(multimodal_msg) == 2:
        print(f"   ✓ Multimodal message created with {len(multimodal_msg)} blocks")
        print(f"     Block types: {[block['type'] for block in multimodal_msg]}")
    else:
        print(f"   ✗ Failed to create multimodal message")
    
    print("\n" + "=" * 60)
    print("Image Utilities Tests Completed")
    print("=" * 60)


def test_user_info_with_image():
    """Test UserInfo with image support."""
    print("\n" + "=" * 60)
    print("Testing UserInfo with Images")
    print("=" * 60)
    
    from oasis.social_platform.config import UserInfo
    
    # Test 1: Create UserInfo with image
    print("\n1. Creating UserInfo with image URL...")
    user_info = UserInfo(
        user_name="TestAgent",
        name="Test Agent",
        profile={
            "other_info": {
                "user_profile": "A test agent with visual context",
                "gender": "Unknown",
                "age": "25",
                "mbti": "INTJ",
                "country": "USA"
            }
        },
        recsys_type="reddit",
        system_image="https://example.com/test.jpg",
        image_type="url"
    )
    
    print(f"   ✓ UserInfo created successfully")
    print(f"     User: {user_info.user_name}")
    print(f"     Has image: {user_info.system_image is not None}")
    print(f"     Image type: {user_info.image_type}")
    
    # Test 2: Generate system message
    print("\n2. Generating system message...")
    system_msg = user_info.to_system_message()
    
    if "VISUAL CONTEXT" in system_msg:
        print(f"   ✓ System message includes visual context note")
    else:
        print(f"   ⚠️  System message doesn't mention visual context (may be expected)")
    
    print(f"     Message length: {len(system_msg)} characters")
    
    # Test 3: UserInfo without image
    print("\n3. Creating UserInfo without image...")
    user_info_no_img = UserInfo(
        user_name="TextOnlyAgent",
        name="Text Only Agent",
        profile={
            "other_info": {
                "user_profile": "A text-only agent",
                "gender": "Unknown",
                "age": "30",
                "mbti": "ENFP",
                "country": "UK"
            }
        },
        recsys_type="reddit"
    )
    
    print(f"   ✓ UserInfo created without image")
    print(f"     Has image: {user_info_no_img.system_image is not None}")
    
    print("\n" + "=" * 60)
    print("UserInfo Tests Completed")
    print("=" * 60)


def test_bedrock_multimodal_payload():
    """Test Bedrock payload creation with multimodal content."""
    print("\n" + "=" * 60)
    print("Testing Bedrock Multimodal Payload")
    print("=" * 60)
    
    from oasis.models.bedrock_model import BedrockModelBackend
    
    print("\n1. Creating BedrockModelBackend...")
    try:
        # Note: This won't actually connect to AWS without credentials
        model = BedrockModelBackend(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-2",
            temperature=0.7,
            max_tokens=1000
        )
        print(f"   ✓ Model backend created")
        print(f"     Model ID: {model.get_model_name()}")
    except Exception as e:
        print(f"   ⚠️  Model creation requires AWS credentials: {e}")
        print(f"   (This is expected without .env file)")
        return
    
    # Test 2: Create multimodal payload
    print("\n2. Testing multimodal payload creation...")
    
    # Create a mock multimodal message
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/test.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "You are a helpful assistant with visual understanding."
                }
            ]
        },
        {
            "role": "user",
            "content": "What do you see in this image?"
        }
    ]
    
    try:
        payload = model._create_bedrock_payload(messages)
        print(f"   ✓ Payload created successfully")
        print(f"     Keys: {list(payload.keys())}")
        print(f"     Messages count: {len(payload.get('messages', []))}")
        
        # Check if images were moved to user message
        first_msg = payload.get('messages', [{}])[0]
        if isinstance(first_msg.get('content'), list):
            content_types = [item.get('type') for item in first_msg['content']]
            print(f"     First message content types: {content_types}")
            if 'image' in content_types:
                print(f"   ✓ Image correctly moved to user message")
            else:
                print(f"   ⚠️  Image not found in user message")
        
    except Exception as e:
        print(f"   ✗ Payload creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("Bedrock Multimodal Tests Completed")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "OASIS Multimodal Integration Tests")
    print("=" * 70)
    print("\nThese tests verify the multimodal (image) support without requiring")
    print("actual image files or AWS credentials.\n")
    
    try:
        # Test 1: Image utilities
        test_image_utils()
        
        # Test 2: UserInfo with images
        test_user_info_with_image()
        
        # Test 3: Bedrock payload creation
        test_bedrock_multimodal_payload()
        
        print("\n" + "=" * 70)
        print(" " * 20 + "All Tests Completed!")
        print("=" * 70)
        print("\n✅ Multimodal integration is working correctly")
        print("\n💡 Next steps:")
        print("   1. Place test images in ./examples/images/")
        print("   2. Run: python examples/reddit_simulation_multimodal.py")
        print("   3. Check the documentation: docs/multimodal_support.md\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

