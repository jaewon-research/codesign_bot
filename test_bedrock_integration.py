#!/usr/bin/env python3
"""
Test Bedrock Integration with OASIS
==================================

This script tests the Bedrock integration to ensure it works correctly.
"""

import asyncio
import os
import sys
import logging

# Add the oasis directory to the path
sys.path.append('/Users/robertgiometti/School/Research/CommunityBot/codesign_bot/oasis')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bedrock_imports():
    """Test that Bedrock modules can be imported."""
    try:
        from oasis.models import BedrockModelBackend, BedrockModelFactory
        print("✅ Bedrock imports successful")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Bedrock modules: {e}")
        return False

def test_bedrock_model_creation():
    """Test creating a Bedrock model."""
    try:
        from oasis.models import BedrockModelFactory
        
        # Test model creation (without actually calling AWS)
        model = BedrockModelFactory.create(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",
            temperature=0.7,
            max_tokens=4096
        )
        
        print("✅ Bedrock model creation successful")
        print(f"   Model ID: {model.model_id}")
        print(f"   Region: {model.region_name}")
        print(f"   Temperature: {model.temperature}")
        return True
    except Exception as e:
        print(f"❌ Failed to create Bedrock model: {e}")
        return False

def test_bedrock_factory_methods():
    """Test Bedrock factory methods."""
    try:
        from oasis.models import BedrockModelFactory
        
        # Test different factory methods
        claude_3_5_sonnet = BedrockModelFactory.create_claude_3_5_sonnet()
        claude_sonnet = BedrockModelFactory.create_claude_3_sonnet()
        claude_haiku = BedrockModelFactory.create_claude_3_haiku()
        titan_large = BedrockModelFactory.create_titan_text_large()
        j2_ultra = BedrockModelFactory.create_j2_ultra()
        
        print("✅ Bedrock factory methods successful")
        print(f"   Claude 3.5 Sonnet: {claude_3_5_sonnet.model_id}")
        print(f"   Claude 3 Sonnet: {claude_sonnet.model_id}")
        print(f"   Claude Haiku: {claude_haiku.model_id}")
        print(f"   Titan Large: {titan_large.model_id}")
        print(f"   J2 Ultra: {j2_ultra.model_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to test factory methods: {e}")
        return False

def test_aws_credentials():
    """Test AWS credentials configuration."""
    # Load .env file if it exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"📁 Loading credentials from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"').strip("'")
                    os.environ[key.strip()] = value
    
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing AWS credentials: {missing_vars}")
        print("   Please create a .env file with your AWS credentials")
        return False
    else:
        print("✅ AWS credentials found")
        return True

async def test_bedrock_api_call():
    """Test actual Bedrock API call (requires AWS credentials)."""
    if not test_aws_credentials():
        print("⏭️  Skipping API test (no AWS credentials)")
        return True
    
    try:
        from oasis.models import BedrockModelFactory
        from camel.messages import BaseMessage
        
        # Create model
        model = BedrockModelFactory.create_claude_3_haiku(
            region_name="us-east-1",
            temperature=0.7,
            max_tokens=100
        )
        
        # Create test message
        messages = [
            BaseMessage.make_user_message(
                role_name="user",
                content="Hello, this is a test message. Please respond with 'Test successful'."
            )
        ]
        
        # Make API call
        print("🔄 Making Bedrock API call...")
        response = await model.agenerate_response(messages)
        
        print("✅ Bedrock API call successful")
        print(f"   Response: {response['output_messages'][0].content}")
        return True
        
    except Exception as e:
        print(f"❌ Bedrock API call failed: {e}")
        return False

def test_oasis_integration():
    """Test OASIS integration with Bedrock."""
    try:
        from oasis import BedrockModelFactory, ActionType
        
        # Test that Bedrock is available in OASIS
        model = BedrockModelFactory.create_claude_3_sonnet()
        actions = ActionType.get_default_reddit_actions()
        
        print("✅ OASIS integration successful")
        print(f"   Model: {model.get_model_name()}")
        print(f"   Available actions: {len(actions)}")
        return True
        
    except Exception as e:
        print(f"❌ OASIS integration failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Bedrock Integration with OASIS")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_bedrock_imports),
        ("Model Creation Test", test_bedrock_model_creation),
        ("Factory Methods Test", test_bedrock_factory_methods),
        ("AWS Credentials Test", test_aws_credentials),
        ("OASIS Integration Test", test_oasis_integration),
        ("Bedrock API Test", test_bedrock_api_call),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Bedrock integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
