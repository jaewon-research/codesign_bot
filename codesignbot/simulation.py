# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from colorama import Back
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from codesign_platform import CodesignPlatform

from oasis.clock.clock import Clock
# Use custom agent generator with note support
from codesign_agents_generator import generate_codesign_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

from database import (
    create_note_tables,
    create_profile_table,
    create_friendship_tables,
    create_follow_request_table,
    create_notification_table,
    create_simulation_meta_table,
    create_agent_thought_table,
    set_simulation_meta,
    get_simulation_meta,
    timestep_to_unix,
)
from recsys import update_rec_table_filtered
import time
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# Log setup
social_log = logging.getLogger(name="social")
social_log.propagate = False
social_log.setLevel("DEBUG")

file_handler = logging.FileHandler("social.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(stream_handler)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default=DEFAULT_CONFIG_PATH,
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Run in test mode with 16 agents for 5 timesteps.",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
print(f"DATA_DIR: {DATA_DIR}")
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "agents.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "agents_test.csv")
print(f"TEST_CSV_PATH: {TEST_CSV_PATH}")
TEST_TIMESTEPS = 3  # Number of timesteps for test mode

# The human user is defined as user_id = 0
HUMAN_USER_ID = 0

async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    num_timesteps: int = 3,
    simulation_hours: float = 24.0,  # Total duration of simulation in hours
    batch_size: int = 16,  # Agents per batch (1 = sequential, num_agents = fully concurrent)
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    inference_configs: dict[str, Any] | None = None,
    available_actions: list[ActionType] = None,
) -> None:
    if inference_configs is None:
        raise ValueError("inference_configs is required. Please provide a config file with --config_path")

    # Set up database
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create database connection
    db = sqlite3.connect(db_path)
    db_cursor = db.cursor()

    # Add tables specific to codesignbot simulation (IF NOT EXISTS)
    create_note_tables(db, db_cursor)
    create_profile_table(db, db_cursor)
    create_friendship_tables(db, db_cursor)
    create_follow_request_table(db, db_cursor)
    create_notification_table(db, db_cursor)
    create_simulation_meta_table(db, db_cursor)
    create_agent_thought_table(db, db_cursor)
    
    # Calculate simulation timing using Unix timestamps
    simulation_start_time = int(time.time())  # Current Unix timestamp
    seconds_per_timestep = (simulation_hours * 3600) / num_timesteps
    
    # Store simulation metadata
    sim_meta = set_simulation_meta(
        db_cursor, db,
        start_time=simulation_start_time,
        simulation_hours=simulation_hours,
        num_timesteps=num_timesteps
    )
    print(f"📅 Simulation timing: {simulation_hours}h total, {num_timesteps} timesteps", flush=True)
    print(f"   Each timestep = {seconds_per_timestep/3600:.2f} hours ({seconds_per_timestep:.0f} seconds)", flush=True)
    print(f"   Start time: {simulation_start_time} (Unix timestamp)", flush=True)
    
    # Clear previous agent data while preserving human notes (user_id=0)
    # This allows running simulation multiple times without conflicts
    try:
        db_cursor.execute("DELETE FROM user")  # Will be recreated by generate_agents
        db_cursor.execute("DELETE FROM follow")
        db_cursor.execute("DELETE FROM post")
        db_cursor.execute("DELETE FROM rec")
        db_cursor.execute("DELETE FROM trace")
        db_cursor.execute("DELETE FROM connection")
        db_cursor.execute("DELETE FROM friend_request")
        # Keep notes - they contain human posts that should persist
        db.commit()
        social_log.info("Cleared agent data, preserved notes")
    except Exception as e:
        social_log.warning(f"Could not clear some tables (may not exist): {e}")

    # Set up infrastructure
    clock = Clock(k=clock_factor)
    clock.time_step = simulation_start_time  # Initialize with start Unix timestamp
    channel = Channel()

    platform = CodesignPlatform(
        db_path=db_path,
        channel=channel,
        sandbox_clock=clock,
        start_time=simulation_start_time,  # Use Unix timestamp
        recsys_type=recsys_type,
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )
    
    # Set database connection for storing agent thoughts
    CodesignPlatform.set_db_connection(db, db_cursor)

    # Start simulation event loop
    simulation_task = asyncio.create_task(platform.running())

    # Set up LLM model
    print("🧠 Setting up LLM model...", flush=True)
    if inference_configs["model_type"][:3] == "gpt":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType(inference_configs["model_type"]),
        )
    print(f"✅ LLM model ready: {inference_configs['model_type']}", flush=True)

    print("🔄 Generating agents...", flush=True)
    agent_graph = await generate_codesign_agents(agent_info_path=csv_path,
                                        channel=channel,
                                        start_time=simulation_start_time,
                                        model=model,
                                        recsys_type=recsys_type,
                                        available_actions=available_actions,
                                        twitter=platform)
    print(f"✅ Generated {agent_graph.get_num_nodes()} agents", flush=True)
    # agent_graph.visualize("initial_social_graph.png")

    # Simulation starts at 1PM
    start_hour = 13

    # Main simulation loop
    print(f"🚀 Starting main loop with {num_timesteps} timesteps...", flush=True)
    for timestep in range(1, num_timesteps + 1):
        # Set clock to Unix timestamp for this timestep
        current_unix_time = simulation_start_time + int(timestep * seconds_per_timestep)
        clock.time_step = current_unix_time
        
        db_file = db_path.split("/")[-1]
        hours_elapsed = (timestep * seconds_per_timestep) / 3600
        print(f"⏱️ Timestep {timestep}/{num_timesteps} (t={current_unix_time}, +{hours_elapsed:.1f}h)", flush=True)

        # Custom twitter-style recsys with connection degree filtering
        print(f"  📊 Updating recommendations...", flush=True)
        await update_rec_table_filtered(
            platform,
            max_connection_degree=3,
            current_time=clock.time_step,
        )

        # Calculate the time window for this timestep
        # Previous timestep end = current timestep start
        timestep_start_unix = simulation_start_time + int((timestep - 1) * seconds_per_timestep)
        timestep_end_unix = simulation_start_time + int(timestep * seconds_per_timestep)
        
        # 0.05 * timestep here means 3 minutes / timestep (for activity threshold)
        simulation_time_hour = start_hour + 0.05 * timestep
        
        # Collect all agents that will act this timestep, with sampled action times
        active_agents_with_times = []
        for node_id, agent in agent_graph.get_agents():
            # Skip the human user in the simulation loop
            # The human user interacts through the UI
            if node_id == HUMAN_USER_ID:
                continue

            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()

                other_info = agent.user_info.profile.get("other_info", {})
                active_threshold = other_info.get("active_threshold", [1.0] * 24)
                threshold = active_threshold[int(simulation_time_hour % 24)]
                if agent_ac_prob < threshold:
                    # Sample a random timestamp within this timestep's interval
                    sampled_time = random.randint(timestep_start_unix, timestep_end_unix)
                    active_agents_with_times.append((sampled_time, node_id, agent))
            else:
                await agent.perform_action_by_hci()

        # Sort agents by their sampled time (earlier times first)
        active_agents_with_times.sort(key=lambda x: x[0])
        
        total_active = len(active_agents_with_times)
        num_batches = (total_active + batch_size - 1) // batch_size if batch_size > 0 else 1
        
        print(f"  🤖 {total_active} agents taking action in {num_batches} batch(es) of {batch_size}...", flush=True)
        
        # Process agents in batches, ordered by sampled time
        # Each agent gets their individual sampled timestamp via CodesignPlatform
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_active)
            batch = active_agents_with_times[batch_start:batch_end]
            
            if batch:
                # Set individual timestamps for each agent in this batch
                # The platform will use these when creating notes/comments
                for sampled_time, node_id, _ in batch:
                    CodesignPlatform.set_agent_timestamp(node_id, sampled_time)
                
                # Also set clock to batch max for any fallback/legacy code
                batch_max_time = max(item[0] for item in batch)
                clock.time_step = batch_max_time
                
                # Run this batch concurrently - each agent uses their assigned timestamp
                batch_tasks = [agent.perform_action_by_llm() for _, _, agent in batch]
                await asyncio.gather(*batch_tasks)
                
                if num_batches > 1:
                    # Show time range for this batch
                    batch_min_time = min(item[0] for item in batch)
                    time_start = (batch_min_time - timestep_start_unix) / 3600
                    time_end = (batch_max_time - timestep_start_unix) / 3600
                    print(f"    ✓ Batch {batch_idx + 1}/{num_batches} complete ({len(batch)} agents, +{time_start:.1f}h to +{time_end:.1f}h)", flush=True)
        
        # Clear agent timestamps at end of timestep
        CodesignPlatform.clear_agent_timestamps()
        
        print(f"  ✅ Timestep {timestep} complete", flush=True)
        # agent_graph.visualize(f"timestep_{timestep}_social_graph.png")

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await simulation_task


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = yaml.safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        inference_configs = cfg.get("inference")

        if args.test:
            data_params["csv_path"] = TEST_CSV_PATH
            simulation_params["num_timesteps"] = TEST_TIMESTEPS
            print(f"🧪 Running in TEST MODE: {TEST_TIMESTEPS} timesteps, using {TEST_CSV_PATH}")

        asyncio.run(
            running(**data_params,
                    **simulation_params,
                    inference_configs=inference_configs))
    else:
        asyncio.run(running())
    social_log.info("Simulation finished.")
