#!/usr/bin/env python3
"""
OASIS Social Simulation Demo Backend
Provides API endpoints for simulation management and agent-specific data
"""

import sqlite3
import json
import os
import subprocess
import threading
import time
import sys
import asyncio
import logging
import glob
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import random

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,  # Show INFO level and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simulation.log')
    ]
)

# Enable detailed logging for key components
logging.getLogger('oasis.models.bedrock_model').setLevel(logging.INFO)
logging.getLogger('oasis.social_agent.agent').setLevel(logging.INFO)
logging.getLogger('oasis.environment').setLevel(logging.INFO)
logging.getLogger('camel.agents.chat_agent').setLevel(logging.WARNING)  # Reduce CAMEL verbosity

# Create logger for our app
app_logger = logging.getLogger('oasis_app')

# Add OASIS imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import oasis
from oasis import (
    ActionType, 
    AgentGraph, 
    LLMAction,
    SocialAgent,
    UserInfo,
    BedrockModelFactory,
    BedrockModelBackend
)
from oasis.social_platform import Channel

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Path to OASIS simulation database
SIMULATION_DB_PATH = "reddit_bedrock_simulation.db"

# Global simulation state
simulation_state = {
    'running': False,
    'pid': None,
    'start_time': None,
    'logs': [],
    'step_mode': False,
    'paused': False,
    'current_step': 0,
    'total_steps': 0,
    'progress': {
        'current_turn': 0,
        'total_turns': 0,
        'agents_active': 0,
        'posts_created': 0,
        'comments_created': 0
    }
}

class IntegratedSimulation:
    """Integrated simulation runner that works within Flask backend."""
    
    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.model = None
        self.running = False
        self.step_mode = False
        self.paused = False
        self.current_step = 0
        self.total_steps = 0
        
    async def initialize(self):
        """Initialize the simulation environment."""
        try:
            log_message("🚀 Initializing integrated simulation...")
            app_logger.info("=" * 60)
            app_logger.info("🚀 STARTING OASIS SIMULATION INITIALIZATION")
            app_logger.info("=" * 60)
            
            # Check AWS credentials
            log_message("🔐 Checking AWS credentials...")
            if not self._check_aws_credentials():
                log_message("❌ AWS credentials not found")
                return False
            log_message("✅ AWS credentials verified")
            
            # Create Bedrock model
            log_message("🤖 Setting up AWS Bedrock model...")
            app_logger.info("Creating Claude 3.5 Sonnet model with multimodal support")
            self.model = BedrockModelFactory.create_claude_3_5_sonnet(
                region_name="us-east-2",
                temperature=0.7,
                max_tokens=4096
            )
            log_message("✅ Bedrock model created successfully")
            app_logger.info("✅ Bedrock model initialized with Claude 3.5 Sonnet")
            
            # Create agent graph
            log_message("👥 Creating agent graph...")
            app_logger.info("Loading user profiles and creating agents...")
            self.agent_graph = await self._create_agent_graph()
            if not self.agent_graph:
                log_message("❌ Failed to create agent graph")
                return False
            log_message(f"✅ Created agent graph with {len(self.agent_graph.get_agents())} agents")
            app_logger.info(f"✅ Agent graph created with {len(self.agent_graph.get_agents())} agents")
            
            # Create environment
            log_message("🌍 Creating simulation environment...")
            db_path = os.path.abspath("reddit_bedrock_simulation.db")
            if os.path.exists(db_path):
                os.remove(db_path)
                log_message("🗑️ Removed old database")
                app_logger.info("🗑️ Cleaned up previous database")
            
            app_logger.info("Creating OASIS environment with Reddit platform...")
            self.env = oasis.make(
                agent_graph=self.agent_graph,
                platform=oasis.DefaultPlatformType.REDDIT,
                database_path=db_path,
            )
            log_message("✅ Environment created successfully")
            app_logger.info("✅ OASIS environment created")
            
            # Initialize environment
            log_message("🔄 Initializing environment...")
            app_logger.info("Resetting environment state...")
            await self.env.reset()
            log_message("✅ Environment initialized")
            app_logger.info("✅ Environment reset completed")
            
            app_logger.info("=" * 60)
            app_logger.info("🎉 SIMULATION INITIALIZATION COMPLETE")
            app_logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            log_message(f"❌ Failed to initialize simulation: {e}")
            app_logger.error(f"❌ Initialization failed: {e}")
            import traceback
            app_logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def run_step(self):
        """Run a single simulation step."""
        if not self.env:
            log_message("❌ Simulation not initialized")
            app_logger.error("❌ Cannot run step: simulation not initialized")
            return False
        
        try:
            if self.step_mode:
                log_message(f"🔍 Step Mode: Executing step {self.current_step + 1}")
                app_logger.info(f"🔍 STEP MODE: Executing step {self.current_step + 1}")
                self.current_step += 1
            else:
                app_logger.info("🔄 Running simulation step...")
            
            # Run one step of the simulation
            app_logger.info("📝 Calling env.step() - processing all agents...")
            
            # Create LLM actions for all agents
            llm_actions = {
                agent: LLMAction()
                for _, agent in self.env.agent_graph.get_agents()
            }
            app_logger.info(f"🤖 Created LLM actions for {len(llm_actions)} agents")
            
            await self.env.step(llm_actions)
            app_logger.info("✅ env.step() completed successfully")
            
            # Update progress
            update_progress()
            
            if self.step_mode:
                log_message(f"✅ Step {self.current_step} completed")
                app_logger.info(f"✅ Step {self.current_step} completed - pausing for user input")
                self.paused = True  # Pause after each step in step mode
            else:
                app_logger.info("✅ Simulation step completed")
            
            return True
            
        except Exception as e:
            log_message(f"❌ Error in simulation step: {e}")
            app_logger.error(f"❌ Step execution failed: {e}")
            import traceback
            app_logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def run_continuous(self):
        """Run simulation continuously."""
        if not self.env:
            log_message("❌ Simulation not initialized")
            app_logger.error("❌ Cannot run continuous simulation: not initialized")
            return False
        
        try:
            log_message("🚀 Starting continuous simulation...")
            app_logger.info("=" * 60)
            app_logger.info("🚀 STARTING CONTINUOUS SIMULATION")
            app_logger.info("=" * 60)
            self.running = True
            
            # Run simulation steps
            for step in range(5):  # 5 steps total
                if not self.running:
                    app_logger.info("🛑 Simulation stopped by user")
                    break
                
                log_message(f"📝 Running step {step + 1}/5")
                app_logger.info(f"📝 SIMULATION STEP {step + 1}/5")
                app_logger.info("🔄 Processing all agents...")
                
                # Create LLM actions for all agents
                llm_actions = {
                    agent: LLMAction()
                    for _, agent in self.env.agent_graph.get_agents()
                }
                app_logger.info(f"🤖 Created LLM actions for {len(llm_actions)} agents")
                
                await self.env.step(llm_actions)
                app_logger.info(f"✅ Step {step + 1} completed successfully")
                
                update_progress()
                
                # Check if we should pause for step mode
                if self.step_mode and self.paused:
                    log_message("⏸️ Paused for step mode")
                    app_logger.info("⏸️ Paused for step mode - waiting for user input")
                    while self.paused and self.step_mode and self.running:
                        await asyncio.sleep(0.1)
                
                if not self.running:
                    app_logger.info("🛑 Simulation stopped during execution")
                    break
            
            log_message("✅ Simulation completed")
            app_logger.info("=" * 60)
            app_logger.info("🎉 SIMULATION COMPLETED SUCCESSFULLY")
            app_logger.info("=" * 60)
            return True
            
        except Exception as e:
            log_message(f"❌ Error in continuous simulation: {e}")
            app_logger.error(f"❌ Continuous simulation failed: {e}")
            import traceback
            app_logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            self.running = False
            app_logger.info("🔄 Simulation state reset")
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        log_message("🛑 Simulation stopped")
    
    def pause(self):
        """Pause the simulation."""
        self.paused = True
        log_message("⏸️ Simulation paused")
    
    def resume(self):
        """Resume the simulation."""
        self.paused = False
        log_message("▶️ Simulation resumed")
    
    def _check_aws_credentials(self):
        """Check if AWS credentials are available."""
        try:
            import boto3
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                log_message("✅ AWS credentials found")
                return True
            else:
                log_message("❌ No AWS credentials found")
                return False
        except Exception as e:
            log_message(f"❌ Error checking AWS credentials: {e}")
            return False
    
    async def _create_agent_graph(self):
        """Create the agent graph with multimodal support."""
        try:
            # Load user data
            user_data_path = "data/reddit/user_data_36.json"
            if not os.path.exists(user_data_path):
                log_message(f"❌ User data not found at {user_data_path}")
                return None
            
            with open(user_data_path, 'r') as f:
                agent_info = json.load(f)
            
            log_message(f"📊 Loaded {len(agent_info)} user profiles")
            
            # Load system images
            system_images = self._load_system_images()
            
            # Create agent graph
            agent_graph = AgentGraph()
            available_actions = ActionType.get_default_reddit_actions()
            
            # Create agents
            for i, user_data in enumerate(agent_info):
                try:
                    # Create user info
                    user_info = UserInfo(
                        name=user_data["username"],
                        description=user_data["bio"],
                        profile=user_data,
                        recsys_type="reddit",
                        system_image=", ".join(system_images) if system_images else None,
                        image_type="file" if system_images else None
                    )
                    
                    # Create agent
                    agent = SocialAgent(
                        agent_id=i,
                        user_info=user_info,
                        model=self.model,
                        available_actions=available_actions
                    )
                    
                    agent_graph.add_agent(agent)
                    
                except Exception as e:
                    log_message(f"⚠️ Failed to create agent {i}: {e}")
                    continue
            
            log_message(f"✅ Created {len(agent_graph.get_agents())} agents")
            return agent_graph
            
        except Exception as e:
            log_message(f"❌ Error creating agent graph: {e}")
            return None
    
    def _load_system_images(self):
        """Load system images for multimodal support."""
        try:
            images_dir = "data/reddit/images"
            if not os.path.exists(images_dir):
                return []
            
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            
            if image_files:
                log_message(f"📸 Loaded {len(image_files)} system images")
            else:
                log_message("⚠️ No system images found")
            
            return image_files
            
        except Exception as e:
            log_message(f"⚠️ Error loading system images: {e}")
            return []

# Global simulation instance
simulation = IntegratedSimulation()

async def run_integrated_simulation():
    """Run the integrated simulation."""
    global simulation_state
    
    try:
        simulation_state['running'] = True
        simulation_state['start_time'] = datetime.now()
        
        # Initialize simulation
        if not await simulation.initialize():
            log_message("❌ Failed to initialize simulation")
            return
        
        # Run simulation
        if simulation_state['step_mode']:
            # Step mode - wait for user input
            simulation.step_mode = True
            simulation.paused = True
            log_message("🔍 Step mode enabled - waiting for user input")
        else:
            # Continuous mode
            await simulation.run_continuous()
            
    except Exception as e:
        log_message(f"❌ Error running simulation: {e}")
    finally:
        simulation_state['running'] = False
        simulation_state['pid'] = None
        log_message("Simulation stopped")

def get_db_connection():
    """Get database connection."""
    if os.path.exists(SIMULATION_DB_PATH):
        return sqlite3.connect(SIMULATION_DB_PATH)
    return None

def log_message(message):
    """Add a message to the simulation logs."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    simulation_state['logs'].append(log_entry)
    app_logger.info(message)  # Use proper logger
    # Keep only last 100 log entries
    if len(simulation_state['logs']) > 100:
        simulation_state['logs'] = simulation_state['logs'][-100:]

def update_progress():
    """Update simulation progress from database."""
    try:
        if not os.path.exists(SIMULATION_DB_PATH):
            return
        
        conn = sqlite3.connect(SIMULATION_DB_PATH)
        cursor = conn.cursor()
        
        # Get post count
        cursor.execute("SELECT COUNT(*) FROM post")
        posts_count = cursor.fetchone()[0]
        
        # Get comment count
        cursor.execute("SELECT COUNT(*) FROM comment")
        comments_count = cursor.fetchone()[0]
        
        # Get user count
        cursor.execute("SELECT COUNT(*) FROM user")
        users_count = cursor.fetchone()[0]
        
        conn.close()
        
        simulation_state['progress'].update({
            'posts_created': posts_count,
            'comments_created': comments_count,
            'agents_active': users_count
        })
        
    except Exception as e:
        log_message(f"Error updating progress: {e}")

def run_simulation():
    """Run the OASIS simulation in a separate process."""
    global simulation_state
    
    try:
        log_message("Starting OASIS simulation...")
        
        # Get the simulation script path
        script_path = os.path.abspath("examples/reddit_simulation_bedrock.py")
        
        if not os.path.exists(script_path):
            log_message(f"Error: Simulation script not found at {script_path}")
            return
        
        # Run the simulation
        process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        simulation_state['pid'] = process.pid
        simulation_state['start_time'] = datetime.now()
        simulation_state['running'] = True
        
        log_message(f"Simulation started with PID: {process.pid}")
        
        # Monitor the process
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                log_message(line.strip())
                # Update progress periodically
                update_progress()
            time.sleep(0.1)
        
        # Process finished
        return_code = process.returncode
        if return_code == 0:
            log_message("Simulation completed successfully!")
        else:
            log_message(f"Simulation ended with error code: {return_code}")
            
    except Exception as e:
        log_message(f"Error running simulation: {e}")
    finally:
        simulation_state['running'] = False
        simulation_state['pid'] = None
        log_message("Simulation stopped")

def run_simulation_step():
    """Run the OASIS simulation in step mode."""
    global simulation_state
    
    try:
        log_message("Starting OASIS simulation in step mode...")
        
        # Get the simulation script path
        script_path = os.path.abspath("examples/reddit_simulation_bedrock.py")
        
        if not os.path.exists(script_path):
            log_message(f"Error: Simulation script not found at {script_path}")
            return
        
        # Run the simulation with step mode
        process = subprocess.Popen(
            [sys.executable, script_path, "--step-mode"],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        simulation_state['pid'] = process.pid
        simulation_state['start_time'] = datetime.now()
        simulation_state['running'] = True
        
        log_message(f"Step-mode simulation started with PID: {process.pid}")
        
        # Monitor the process with step control
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                log_message(line.strip())
                # Update progress periodically
                update_progress()
                
                # Check if we should pause for step mode
                if simulation_state['step_mode'] and simulation_state['paused']:
                    log_message("Step mode: Pausing for user input...")
                    # Wait until user clicks "Next Step"
                    while simulation_state['paused'] and simulation_state['step_mode']:
                        time.sleep(0.1)
                    
                    if not simulation_state['step_mode']:
                        break
                    
                    simulation_state['current_step'] += 1
                    log_message(f"Step mode: Continuing to step {simulation_state['current_step']}")
            
            time.sleep(0.1)
        
        # Process finished
        return_code = process.returncode
        if return_code == 0:
            log_message("Step-mode simulation completed successfully!")
        else:
            log_message(f"Step-mode simulation ended with error code: {return_code}")
            
    except Exception as e:
        log_message(f"Error running step-mode simulation: {e}")
    finally:
        simulation_state['running'] = False
        simulation_state['pid'] = None
        log_message("Step-mode simulation stopped")

def get_simulation_stats():
    """Get simulation statistics."""
    conn = get_db_connection()
    if not conn:
        return {
            "totalAgents": 36,
            "activeAgents": 28,
            "totalPosts": 156,
            "totalComments": 423,
            "totalLikes": 892
        }
    
    try:
        cursor = conn.cursor()
        
        # Get user count
        cursor.execute("SELECT COUNT(*) FROM user")
        total_agents = cursor.fetchone()[0]
        
        # Get posts count
        cursor.execute("SELECT COUNT(*) FROM post")
        total_posts = cursor.fetchone()[0]
        
        # Get comments count
        cursor.execute("SELECT COUNT(*) FROM comment")
        total_comments = cursor.fetchone()[0]
        
        # Get likes count
        cursor.execute("SELECT COUNT(*) FROM like")
        total_likes = cursor.fetchone()[0]
        
        return {
            "totalAgents": total_agents,
            "activeAgents": total_agents,  # Assume all are active
            "totalPosts": total_posts,
            "totalComments": total_comments,
            "totalLikes": total_likes
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            "totalAgents": 36,
            "activeAgents": 28,
            "totalPosts": 156,
            "totalComments": 423,
            "totalLikes": 892
        }
    finally:
        conn.close()

def get_posts_with_comments(limit=10):
    """Get recent posts with their comments."""
    conn = get_db_connection()
    if not conn:
        # Return mock data if no database
        return get_mock_posts()
    
    try:
        cursor = conn.cursor()
        
        # Get recent posts
        cursor.execute("""
            SELECT p.id, p.content, p.created_at, u.username, u.name
            FROM post p
            JOIN user u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            LIMIT ?
        """, (limit,))
        
        posts_data = cursor.fetchall()
        posts = []
        
        for post_id, content, created_at, username, name in posts_data:
            # Get comments for this post
            cursor.execute("""
                SELECT c.id, c.content, c.created_at, u.username
                FROM comment c
                JOIN user u ON c.user_id = u.id
                WHERE c.post_id = ?
                ORDER BY c.created_at ASC
                LIMIT 5
            """, (post_id,))
            
            comments_data = cursor.fetchall()
            comments = []
            
            for comment_id, comment_content, comment_created, comment_username in comments_data:
                comments.append({
                    "id": comment_id,
                    "author": comment_username,
                    "content": comment_content,
                    "timestamp": comment_created
                })
            
            # Get like count
            cursor.execute("SELECT COUNT(*) FROM like WHERE post_id = ?", (post_id,))
            like_count = cursor.fetchone()[0]
            
            posts.append({
                "id": post_id,
                "author": {
                    "id": post_id,  # Simplified
                    "username": username,
                    "name": name,
                    "avatar": name[:2].upper() if name else username[:2].upper()
                },
                "content": content,
                "timestamp": created_at,
                "likes": like_count,
                "comments": len(comments),
                "shares": random.randint(0, 10),
                "isLiked": False,
                "comments_list": comments
            })
        
        return posts
        
    except Exception as e:
        print(f"Error getting posts: {e}")
        return get_mock_posts()
    finally:
        conn.close()

def get_mock_posts():
    """Return mock posts for demonstration."""
    return [
        {
            "id": 1,
            "author": {
                "id": 1,
                "username": "millerhospitality",
                "name": "James Miller",
                "avatar": "JM"
            },
            "content": "Just discovered this amazing new AI tool for hospitality management. The way it can predict guest preferences is mind-blowing! Anyone else in the industry experimenting with AI?",
            "timestamp": datetime.now() - timedelta(minutes=15),
            "likes": 23,
            "comments": 8,
            "shares": 3,
            "isLiked": False,
            "comments_list": [
                {
                    "id": 1,
                    "author": "emma_logistics_guru",
                    "content": "Fascinating! We're using similar tech in logistics. The predictive capabilities are game-changing.",
                    "timestamp": datetime.now() - timedelta(minutes=12)
                }
            ]
        },
        {
            "id": 2,
            "author": {
                "id": 2,
                "username": "emma_logistics_guru",
                "name": "Emma Hayes",
                "avatar": "EH"
            },
            "content": "The future of transportation is here! Just attended a conference on autonomous logistics. The efficiency gains are incredible - we're talking 40% reduction in delivery times.",
            "timestamp": datetime.now() - timedelta(minutes=45),
            "likes": 67,
            "comments": 15,
            "shares": 12,
            "isLiked": True,
            "comments_list": [
                {
                    "id": 2,
                    "author": "biz_mind45",
                    "content": "Great insights! The job market evolution is definitely something to watch.",
                    "timestamp": datetime.now() - timedelta(minutes=40)
                }
            ]
        }
    ]

# API Routes
@app.route('/api/simulation/<simulation_id>')
def get_simulation_data(simulation_id):
    """Get simulation data including posts and statistics."""
    try:
        stats = get_simulation_stats()
        posts = get_posts_with_comments()
        
        return jsonify({
            "simulation": {
                "id": simulation_id,
                "name": f"Reddit Simulation with Multimodal Agents",
                "status": "running",
                "startTime": (datetime.now() - timedelta(hours=1)).isoformat(),
                **stats
            },
            "posts": posts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/<simulation_id>/posts')
def get_posts(simulation_id):
    """Get posts for a specific simulation."""
    try:
        posts = get_posts_with_comments()
        return jsonify({"posts": posts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/<simulation_id>/stats')
def get_stats(simulation_id):
    """Get simulation statistics."""
    try:
        stats = get_simulation_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# Simulation Management Endpoints
@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get current simulation status."""
    try:
        update_progress()
        
        # Sync with integrated simulation
        simulation_state['running'] = simulation.running
        simulation_state['step_mode'] = simulation.step_mode
        simulation_state['paused'] = simulation.paused
        simulation_state['current_step'] = simulation.current_step
        simulation_state['total_steps'] = simulation.total_steps
        
        return jsonify({
            'running': simulation_state['running'],
            'pid': simulation_state['pid'],
            'start_time': simulation_state['start_time'].isoformat() if simulation_state['start_time'] else None,
            'step_mode': simulation_state['step_mode'],
            'paused': simulation_state['paused'],
            'current_step': simulation_state['current_step'],
            'total_steps': simulation_state['total_steps'],
            'progress': simulation_state['progress'],
            'logs': simulation_state['logs'][-20:]  # Last 20 log entries
        })
    except Exception as e:
        log_message(f"❌ Error in status endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start the OASIS simulation."""
    try:
        if simulation_state['running']:
            return jsonify({'error': 'Simulation is already running'}), 400
        
        # Start simulation in background thread
        def run_async_simulation():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_integrated_simulation())
            loop.close()
        
        thread = threading.Thread(target=run_async_simulation)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Simulation starting...'})
    except Exception as e:
        log_message(f"❌ Error starting simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop the running simulation."""
    if not simulation_state['running']:
        return jsonify({'error': 'No simulation is running'}), 400
    
    try:
        simulation.stop()
        simulation_state['running'] = False
        simulation_state['pid'] = None
        log_message("Simulation stopped by user")
        
        return jsonify({'message': 'Simulation stopped'})
    except Exception as e:
        return jsonify({'error': f'Error stopping simulation: {e}'}), 500

@app.route('/api/simulation/logs', methods=['GET'])
def get_simulation_logs():
    """Get simulation logs."""
    return jsonify({
        'logs': simulation_state['logs']
    })

@app.route('/api/simulation/progress', methods=['GET'])
def get_simulation_progress():
    """Get detailed simulation progress."""
    update_progress()
    return jsonify(simulation_state['progress'])

# Step-by-step simulation endpoints
@app.route('/api/simulation/step/start', methods=['POST'])
def start_step_mode():
    """Start simulation in step mode."""
    if simulation_state['running']:
        return jsonify({'error': 'Simulation is already running'}), 400
    
    simulation_state['step_mode'] = True
    simulation_state['paused'] = True
    simulation_state['current_step'] = 0
    simulation_state['total_steps'] = 0
    
    log_message("Step mode enabled - simulation will pause after each step")
    return jsonify({'message': 'Step mode enabled'})

@app.route('/api/simulation/step/next', methods=['POST'])
def next_step():
    """Execute next step in step mode."""
    if not simulation_state['step_mode']:
        return jsonify({'error': 'Not in step mode'}), 400
    
    if not simulation_state['running']:
        # Start simulation if not running
        def run_async_step():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_integrated_simulation())
            loop.close()
        
        thread = threading.Thread(target=run_async_step)
        thread.daemon = True
        thread.start()
        return jsonify({'message': 'Starting simulation in step mode'})
    
    # Execute next step
    def run_async_step():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(simulation.run_step())
        loop.close()
    
    thread = threading.Thread(target=run_async_step)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'Executing step {simulation_state["current_step"] + 1}'})

@app.route('/api/simulation/step/pause', methods=['POST'])
def pause_step():
    """Pause simulation in step mode."""
    if not simulation_state['step_mode']:
        return jsonify({'error': 'Not in step mode'}), 400
    
    simulation.pause()
    simulation_state['paused'] = True
    log_message("Simulation paused - waiting for next step")
    return jsonify({'message': 'Simulation paused'})

@app.route('/api/simulation/step/resume', methods=['POST'])
def resume_step():
    """Resume simulation in step mode."""
    if not simulation_state['step_mode']:
        return jsonify({'error': 'Not in step mode'}), 400
    
    simulation.resume()
    simulation_state['paused'] = False
    log_message("Simulation resumed")
    return jsonify({'message': 'Simulation resumed'})

@app.route('/api/simulation/step/stop', methods=['POST'])
def stop_step_mode():
    """Stop step mode and return to normal mode."""
    simulation_state['step_mode'] = False
    simulation_state['paused'] = False
    simulation_state['current_step'] = 0
    simulation_state['total_steps'] = 0
    
    if simulation_state['running']:
        # Stop the running simulation
        try:
            if simulation_state['pid']:
                os.kill(simulation_state['pid'], 9)
                log_message("Step mode stopped - simulation terminated")
        except Exception as e:
            log_message(f"Error stopping simulation: {e}")
        
        simulation_state['running'] = False
        simulation_state['pid'] = None
    
    return jsonify({'message': 'Step mode disabled'})

if __name__ == '__main__':
    print("🚀 Starting OASIS Social Simulation Demo Backend")
    print(f"📊 Looking for simulation database at: {SIMULATION_DB_PATH}")
    
    if os.path.exists(SIMULATION_DB_PATH):
        print("✅ Found simulation database")
    else:
        print("⚠️  No simulation database found, using mock data")
    
    print("🌐 Backend running on http://localhost:5000")
    print("📱 Frontend should be running on http://localhost:3000")
    print("🎮 Available simulation endpoints:")
    print("   GET  /api/simulation/status     - Get simulation status")
    print("   POST /api/simulation/start       - Start simulation")
    print("   POST /api/simulation/stop        - Stop simulation")
    print("   GET  /api/simulation/logs       - Get simulation logs")
    print("   GET  /api/simulation/progress    - Get progress")
    
    app.run(debug=True, port=5000)
