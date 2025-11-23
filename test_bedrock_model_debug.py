#!/usr/bin/env python3
"""
Debug script to test BedrockModelBackend count_tokens_from_messages method.
"""

import sys
import os
sys.path.append('/Users/robertgiometti/School/Research/CommunityBot/codesign_bot/oasis')

from oasis.models import BedrockModelFactory
from camel.messages import BaseMessage

def test_bedrock_model_methods():
    """Test BedrockModelBackend methods."""
    print("🧪 Testing BedrockModelBackend methods...")
    
    try:
        # Create a Bedrock model
        model = BedrockModelFactory.create_claude_3_5_sonnet(
            region_name="us-east-1",
            temperature=0.7,
            max_tokens=4096
        )
        
        print(f"✅ Model created: {type(model)}")
        print(f"✅ Model ID: {model.model_id}")
        
        # Test if count_tokens_from_messages exists
        if hasattr(model, 'count_tokens_from_messages'):
            print("✅ count_tokens_from_messages method exists")
            print(f"   Method type: {type(model.count_tokens_from_messages)}")
            print(f"   Method callable: {callable(model.count_tokens_from_messages)}")
        else:
            print("❌ count_tokens_from_messages method not found")
            print(f"   Available methods: {[method for method in dir(model) if not method.startswith('_')]}")
            return False
        
        # Test the method with sample messages
        test_messages = [
            BaseMessage.make_user_message(role_name="user", content="Hello, world!"),
            BaseMessage.make_assistant_message(role_name="assistant", content="Hi there!")
        ]
        
        try:
            token_count = model.count_tokens_from_messages(test_messages)
            print(f"✅ Token count result: {token_count}")
            return True
        except Exception as e:
            print(f"❌ Error calling count_tokens_from_messages: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bedrock_model_methods()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
