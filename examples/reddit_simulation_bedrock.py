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
Reddit Simulation with AWS Bedrock
==================================

This script demonstrates how to use AWS Bedrock with OASIS for Reddit simulations.
It shows how to configure different Bedrock models and run social media simulations.

Requirements:
- pip install boto3
- AWS credentials configured
- Bedrock access enabled in your AWS account

Usage:
    python reddit_simulation_bedrock.py
"""

import asyncio
import os
import sys
import logging
from typing import List, Dict, Any

# Add the oasis directory to the path
sys.path.append('/Users/robertgiometti/School/Research/CommunityBot/codesign_bot/oasis')

# Configure logging with cleaner output
logging.basicConfig(
    level=logging.WARNING,  # Reduce default noise
    format='%(levelname)s - %(message)s'
)

# Enable INFO logging only for key components
logging.getLogger('oasis.models.bedrock_model').setLevel(logging.INFO)
logging.getLogger('oasis.social_agent.agent').setLevel(logging.INFO)  # Show agent actions
logging.getLogger('camel.agents.chat_agent').setLevel(logging.WARNING)  # Reduce CAMEL verbosity
logging.getLogger('oasis.environment').setLevel(logging.INFO)

import oasis
from oasis import (
    ActionType, 
    AgentGraph, 
    LLMAction, 
    ManualAction, 
    SocialAgent, 
    UserInfo,
    BedrockModelFactory,
    generate_reddit_agent_graph
)


def check_aws_credentials():
    """Check if AWS credentials are properly configured."""
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
        print(f"❌ Missing AWS credentials: {missing_vars}")
        print("Please set the following in your .env file:")
        print("  AWS_ACCESS_KEY_ID=your_access_key")
        print("  AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("  AWS_DEFAULT_REGION=us-east-1")
        return False
    
    print("✅ AWS credentials found")
    return True


async def run_bedrock_simulation():
    """Run a Reddit simulation using AWS Bedrock."""
    print("🚀 Starting Reddit Simulation with AWS Bedrock")
    print("=" * 60)
    
    # Check AWS credentials
    if not check_aws_credentials():
        return
    
    # Create Bedrock model
    print("🤖 Setting up AWS Bedrock model...")
    try:
        bedrock_model = BedrockModelFactory.create_claude_3_5_sonnet(
            region_name="us-east-2",  # Change from us-east-1 to us-east-2
            temperature=0.7,
            max_tokens=4096
        )
        print("✅ Bedrock model created successfully")
    except Exception as e:
        print(f"❌ Failed to create Bedrock model: {e}")
        return
    
    # Define available actions
    available_actions = ActionType.get_default_reddit_actions()
    print(f"📋 Available actions: {[action.value for action in available_actions]}")
    
    # Create agent graph
    print("🔗 Creating agent graph...")
    try:
        agent_graph = await generate_reddit_agent_graph(
            profile_path="./data/reddit/user_data_36.json",
            model=bedrock_model,
            available_actions=available_actions,
        )
        print(f"✅ Agent graph created with {len(agent_graph.get_agents())} agents")
    except Exception as e:
        print(f"❌ Failed to create agent graph: {e}")
        return
    
    # Create environment
    print("🌍 Creating Reddit environment...")
    db_path = "./reddit_bedrock_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)
    
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
    
    # Run simulation
    print("\n🚀 Starting simulation...")
    try:
        await env.reset()
        print(f"✅ Environment initialized with {len(env.agent_graph.get_agents())} agents")
    except Exception as e:
        print(f"❌ Failed to initialize environment: {e}")
        return
    
    # Simulation steps
    print("\n📝 Running simulation steps...")
    
    # Step 1: Initial posts
    print("\n" + "=" * 80)
    print("📝 STEP 1: Creating initial posts")
    print("=" * 80)
    try:
        initial_posts = {}
        topics = [
            "What's your favorite programming language and why?",
            "Share your most interesting travel experience",
            "What's the best book you've read recently?",
            "Discuss the impact of AI on society",
            "What's your go-to comfort food?"
        ]
        
        num_initial_posts = min(5, len(env.agent_graph.get_agents()))
        print(f"📊 Creating {num_initial_posts} initial posts\n")
        
        for i in range(num_initial_posts):
            agent = env.agent_graph.get_agent(i)
            initial_posts[agent] = ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": topics[i]}
            )
            print(f"   {i+1}. Agent {agent.social_agent_id} ({agent.user_info.user_name}): \"{topics[i][:50]}...\"")
        
        await env.step(initial_posts)
        print("\n" + "=" * 80)
        print("✅ STEP 1 COMPLETED - Initial posts created")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Failed to create initial posts: {e}")
        return
    
    # Step 2: LLM actions with Bedrock (test with fewer agents first)
    print("\n" + "=" * 80)
    print("🤖 STEP 2: Natural agent interactions with Bedrock")
    print("=" * 80)
    print("⏱️  This may take 30-60 seconds as agents call Bedrock API...")
    
    # Start with just 3 agents for testing
    num_test_agents = min(3, len(env.agent_graph.get_agents()))
    print(f"📊 Testing with {num_test_agents} agents")
    
    try:
        import time
        start_time = time.time()
        
        test_agents = [agent for _, agent in list(env.agent_graph.get_agents())[:num_test_agents]]
        
        # Print which agents will act
        print("\n👥 Agents in this round:")
        for i, agent in enumerate(test_agents, 1):
            print(f"   {i}. Agent {agent.social_agent_id} - {agent.user_info.user_name}")
        print()
        
        llm_actions = {
            agent: LLMAction()
            for agent in test_agents
        }
        
        await env.step(llm_actions)
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"✅ STEP 2 COMPLETED in {elapsed:.1f}s ({elapsed/num_test_agents:.1f}s per agent)")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Failed to run LLM actions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Another round of interactions (optional - commented out for testing)
    print("\n⏭️  Step 3: Skipping second round for now (you can uncomment to test more)")
    # Uncomment below to test a second round with all agents
    # print("\n🤖 Step 3: Second round of interactions...")
    # try:
    #     llm_actions_2 = {
    #         agent: LLMAction()
    #         for _, agent in env.agent_graph.get_agents()
    #     }
    #     await env.step(llm_actions_2)
    #     print("✅ Second round completed")
    # except Exception as e:
    #     print(f"❌ Failed to run second round: {e}")
    #     return
    
    # Close environment
    print("\n🏁 Simulation completed!")
    try:
        await env.close()
        print("✅ Environment closed successfully")
    except Exception as e:
        print(f"❌ Failed to close environment: {e}")
        return
    
    print(f"\n📊 Results saved to database: {db_path}")
    print("🔍 You can analyze the results using the database or OASIS analysis tools")
    
    # Print simulation statistics
    print("\n📈 Simulation Statistics:")
    print(f"   • Total agents: {len(env.agent_graph.get_agents())}")
    print(f"   • Database file: {db_path}")
    print(f"   • Model: {bedrock_model.get_model_name()}")
    print(f"   • Region: us-east-1")
    print(f"   • Simulation steps: 3")
    
    return db_path


async def run_multi_model_simulation():
    """Run a simulation with multiple Bedrock models for load balancing."""
    print("🚀 Starting Multi-Model Reddit Simulation with AWS Bedrock")
    print("=" * 70)
    
    # Check AWS credentials
    if not check_aws_credentials():
        return
    
    # Create multiple Bedrock models
    print("🤖 Setting up multiple AWS Bedrock models...")
    try:
        from camel.models import ModelManager
        
        # Create different Bedrock models
        claude_sonnet = BedrockModelFactory.create_claude_3_sonnet(
            region_name="us-east-1",
            temperature=0.7
        )
        
        claude_haiku = BedrockModelFactory.create_claude_3_haiku(
            region_name="us-east-1",
            temperature=0.7
        )
        
        # Create model manager for load balancing
        model_manager = ModelManager(
            models=[claude_sonnet, claude_haiku],
            scheduling_strategy='round_robin'
        )
        
        print("✅ Multiple Bedrock models created successfully")
    except Exception as e:
        print(f"❌ Failed to create multiple models: {e}")
        return
    
    # Rest of the simulation logic would be similar...
    print("📝 Multi-model simulation would continue here...")
    print("   (Implementation similar to single model simulation)")


async def main():
    """Main entry point."""
    print("🌍 OASIS Reddit Simulation with AWS Bedrock")
    print("=" * 50)
    
    try:
        # Run single model simulation
        db_path = await run_bedrock_simulation()
        
        if db_path:
            print(f"\n🎉 Reddit simulation completed successfully!")
            print(f"📁 Check the database at: {db_path}")
            print(f"🔍 You can analyze the results using OASIS analysis tools")
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the simulation
    asyncio.run(main())
