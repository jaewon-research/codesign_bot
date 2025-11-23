#!/usr/bin/env python3
"""
Setup Script for AWS Bedrock Integration with OASIS
==================================================

This script helps you set up AWS Bedrock integration with OASIS.
"""

import os
import sys
import subprocess
import platform

def print_header():
    """Print setup header."""
    print("🚀 OASIS AWS Bedrock Integration Setup")
    print("=" * 50)

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install boto3 and botocore
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "boto3>=1.34.0", "botocore>=1.34.0"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_aws_credentials():
    """Check AWS credentials."""
    print("\n🔐 Checking AWS credentials...")
    
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
        print("\n📋 To set up AWS credentials, create a .env file:")
        print("   AWS_ACCESS_KEY_ID=your_access_key")
        print("   AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("   AWS_DEFAULT_REGION=us-east-1")
        print("\nOr use environment variables:")
        for var in missing_vars:
            print(f"   export {var}=your_value")
        return False
    else:
        print("✅ AWS credentials found")
        return True

def test_bedrock_access():
    """Test Bedrock access."""
    print("\n🧪 Testing Bedrock access...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Create Bedrock client
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Try to list foundation models (this requires bedrock:ListFoundationModels permission)
        try:
            response = bedrock.list_foundation_models()
            print("✅ Bedrock access confirmed")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                print("⚠️  Bedrock access limited (this is normal)")
                print("   Make sure you have enabled Bedrock in your AWS account")
                return True
            else:
                print(f"❌ Bedrock access error: {e}")
                return False
                
    except ImportError:
        print("❌ boto3 not installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_integration_test():
    """Run integration test."""
    print("\n🧪 Running integration test...")
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, "test_bedrock_integration.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Integration test passed")
            return True
        else:
            print("❌ Integration test failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run integration test: {e}")
        return False

def print_next_steps():
    """Print next steps."""
    print("\n🎯 Next Steps")
    print("=" * 20)
    print("1. Enable Bedrock models in AWS Console:")
    print("   - Go to AWS Bedrock console")
    print("   - Enable Claude 3 Sonnet and Haiku models")
    print("   - Request access if needed")
    print("\n2. Run a simulation:")
    print("   python examples/reddit_simulation_bedrock.py")
    print("\n3. Check documentation:")
    print("   docs/bedrock_integration.md")
    print("\n4. Monitor costs:")
    print("   - Set up CloudWatch alarms")
    print("   - Monitor Bedrock usage in AWS Console")

def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check AWS credentials
    credentials_ok = check_aws_credentials()
    
    # Test Bedrock access
    bedrock_ok = test_bedrock_access()
    
    # Run integration test
    test_ok = run_integration_test()
    
    # Print results
    print("\n📊 Setup Summary")
    print("=" * 20)
    print(f"Python Version: {'✅' if check_python_version() else '❌'}")
    print(f"Dependencies: {'✅' if install_dependencies() else '❌'}")
    print(f"AWS Credentials: {'✅' if credentials_ok else '⚠️'}")
    print(f"Bedrock Access: {'✅' if bedrock_ok else '❌'}")
    print(f"Integration Test: {'✅' if test_ok else '❌'}")
    
    if credentials_ok and bedrock_ok and test_ok:
        print("\n🎉 Setup completed successfully!")
        print_next_steps()
        return True
    else:
        print("\n⚠️  Setup completed with warnings")
        print("   Some components may need additional configuration")
        print_next_steps()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
