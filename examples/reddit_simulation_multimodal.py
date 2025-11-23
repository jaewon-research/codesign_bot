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
Reddit Simulation with Multimodal Support (Images)
===================================================

This script demonstrates how to use multimodal input (text + images) with OASIS
for Reddit simulations. It shows how to provide context images to agents at the
start of the simulation.

Requirements:
- pip install boto3
- AWS credentials configured
- Bedrock access enabled in your AWS account
- Claude 3 models (Sonnet, Haiku) support vision

Usage:
    python reddit_simulation_multimodal.py
"""

import asyncio
import json
import os
import sys
import logging
from typing import List, Dict, Any

# Add the oasis directory to the path
sys.path.append('/Users/robertgiometti/School/Research/CommunityBot/codesign_bot/oasis')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

import oasis
from oasis import (
    ActionType, 
    AgentGraph, 
    LLMAction, 
    ManualAction, 
    SocialAgent, 
    UserInfo,
    BedrockModelFactory,
)
from oasis.social_platform import Channel


def check_aws_credentials():
    """Check if AWS credentials are properly configured."""
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
        print(f"❌ Missing AWS credentials: {missing_vars}")
        return False
    
    print("✅ AWS credentials found")
    return True


async def create_agent_with_image(
    agent_id: int,
    name: str,
    profile_description: str,
    image_path: str,
    channel: Channel,
    model,
    available_actions: List[ActionType]
) -> SocialAgent:
    """
    Create a social agent with an image context.
    
    Args:
        agent_id: Unique agent ID
        name: Agent name
        profile_description: Text description of the agent
        image_path: Path to the context image
        channel: Communication channel
        model: LLM model backend
        available_actions: List of available actions
        
    Returns:
        SocialAgent with multimodal context
    """
    # Create user info with image
    user_info = UserInfo(
        user_name=name,
        name=name,
        profile={
            "other_info": {
                "user_profile": profile_description,
                "gender": "Unknown",
                "age": "Unknown",
                "mbti": "Unknown",
                "country": "Unknown"
            }
        },
        recsys_type="reddit",
        system_image=image_path,  # Path to image file
        image_type="file"  # Can be 'file', 'url', or 'base64'
    )
    
    # Create agent
    agent = SocialAgent(
        agent_id=agent_id,
        user_info=user_info,
        channel=channel,
        model=model,
        available_actions=available_actions
    )
    
    # Sign up the agent on the platform
    await agent.env.action.sign_up(
        user_name=name,
        name=name,
        bio=profile_description
    )
    
    return agent


async def run_multimodal_simulation():
    """Run a Reddit simulation with multimodal (image) support."""
    print("🚀 Starting Multimodal Reddit Simulation")
    print("=" * 60)
    
    # Check AWS credentials
    if not check_aws_credentials():
        return
    
    # Create Bedrock model (must use Claude 3 for vision support)
    print("🤖 Setting up AWS Bedrock model (Claude 3.5 Sonnet with vision)...")
    try:
        bedrock_model = BedrockModelFactory.create_claude_3_5_sonnet(
            region_name="us-east-2",
            temperature=0.7,
            max_tokens=4096
        )
        print("✅ Bedrock model created successfully")
    except Exception as e:
        print(f"❌ Failed to create Bedrock model: {e}")
        return
    
    # Define available actions
    available_actions = ActionType.get_default_reddit_actions()
    
    # Create shared channel
    channel = Channel()
    
    # Create agent graph
    print("🔗 Creating agent graph with multimodal agents...")
    agent_graph = AgentGraph()
    
    # Example: Create agents with different context images
    # You can replace these with your actual image paths
    agents_config = [
        {
            "name": "NatureLover",
            "description": "An environmentalist who loves nature and outdoor activities",
            "image": "./examples/images/nature.jpg"  # You'll need to provide this
        },
        {
            "name": "TechEnthusiast",
            "description": "A software engineer passionate about AI and technology",
            "image": "./examples/images/technology.jpg"  # You'll need to provide this
        },
        {
            "name": "ArtCritic",
            "description": "An art historian with expertise in modern art",
            "image": "./examples/images/artwork.jpg"  # You'll need to provide this
        }
    ]
    
    # Check if images exist, otherwise use a placeholder approach
    print("\n📸 Setting up agents with image contexts...")
    for i, config in enumerate(agents_config):
        if os.path.exists(config["image"]):
            print(f"   ✅ {config['name']}: Using image from {config['image']}")
            image_path = config["image"]
        else:
            print(f"   ⚠️  {config['name']}: Image not found, using text-only (no image)")
            image_path = None
        
        # Create user info
        user_info = UserInfo(
            user_name=config["name"],
            name=config["name"],
            profile={
                "other_info": {
                    "user_profile": config["description"],
                    "gender": "Unknown",
                    "age": "Unknown",
                    "mbti": "Unknown",
                    "country": "Unknown"
                }
            },
            recsys_type="reddit",
            system_image=image_path,  # Will be None if image doesn't exist
            image_type="file" if image_path else None
        )
        
        # Create agent
        agent = SocialAgent(
            agent_id=i,
            user_info=user_info,
            channel=channel,
            model=bedrock_model,
            available_actions=available_actions,
            agent_graph=agent_graph
        )
        
        # Add to graph
        agent_graph.add_agent(agent)
    
    print(f"✅ Created {len(agent_graph.get_agents())} agents")
    
    # Create environment
    print("\n🌍 Creating Reddit environment...")
    db_path = "./reddit_multimodal_simulation.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🗑️  Removed old database")
    
    try:
        env = oasis.make(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=db_path,
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return
    
    # Initialize environment
    print("\n🚀 Initializing simulation...")
    try:
        await env.reset()
        print(f"✅ Environment initialized")
    except Exception as e:
        print(f"❌ Failed to initialize environment: {e}")
        return
    
    # Run simulation steps
    print("\n" + "=" * 80)
    print("📝 STEP 1: Creating initial posts")
    print("=" * 80)
    
    try:
        initial_posts = {}
        topics = [
            "What's your thoughts on climate change and environmental conservation?",
            "Discuss the latest developments in artificial intelligence",
            "Share your favorite artwork or art movement",
        ]
        
        for i, (_, agent) in enumerate(list(env.agent_graph.get_agents())[:3]):
            initial_posts[agent] = ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": topics[i]}
            )
            print(f"   {i+1}. {agent.user_info.user_name}: \"{topics[i]}\"")
        
        await env.step(initial_posts)
        print("\n✅ STEP 1 COMPLETED - Initial posts created")
    except Exception as e:
        print(f"❌ Failed to create initial posts: {e}")
        return
    
    # Step 2: LLM-driven interactions
    print("\n" + "=" * 80)
    print("🤖 STEP 2: Agent interactions (with multimodal context)")
    print("=" * 80)
    print("⏱️  Agents will use their image context to inform their responses...")
    
    try:
        llm_actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        
        await env.step(llm_actions)
        print("\n✅ STEP 2 COMPLETED - Agents interacted based on their contexts")
    except Exception as e:
        print(f"❌ Failed to run LLM actions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Close environment
    print("\n🏁 Simulation completed!")
    try:
        await env.close()
        print("✅ Environment closed successfully")
    except Exception as e:
        print(f"❌ Failed to close environment: {e}")
        return
    
    print(f"\n📊 Results saved to database: {db_path}")
    print("🔍 Check the interactions to see how image context influenced agent behavior")
    
    return db_path


async def main():
    """Main entry point."""
    print("🌍 OASIS Multimodal Reddit Simulation")
    print("=" * 50)
    print("\nNOTE: This example demonstrates multimodal support.")
    print("To use images, place your images in ./examples/images/ or modify the paths.")
    print("The simulation will work without images (text-only mode) as well.\n")
    
    try:
        db_path = await run_multimodal_simulation()
        
        if db_path:
            print(f"\n🎉 Multimodal simulation completed successfully!")
            print(f"📁 Database: {db_path}")
            
            # Print usage tips
            print("\n💡 Tips:")
            print("   • Agents with images will have visual context in their system prompt")
            print("   • Claude 3 models process both text and images when making decisions")
            print("   • You can mix agents with and without images in the same simulation")
            print("   • Supported formats: JPG, PNG, GIF, WebP (max 5MB)")
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

