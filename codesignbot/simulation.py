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
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from colorama import Back
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

from database import (
    create_profile_table, create_friendship_tables, create_follow_request_table, create_question_tables,
    create_notification_table
)
from recsys import update_rec_table_filtered
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "agents.csv")


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    inference_configs: dict[str, Any] | None = None,
    available_actions: list[ActionType] = None,
) -> None:
    if inference_configs is None:
        raise ValueError("inference_configs is required. Please provide a config file with --config_path")

    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    if os.path.exists(db_path):
        os.remove(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    start_time = 0
    clock = Clock(k=clock_factor)
    channel = Channel()
    infra = Platform(
        db_path=db_path,
        channel=channel,
        sandbox_clock=clock,
        start_time=start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )

    # Add tables specific to codesignbot simulation
    create_profile_table(infra.db, infra.db_cursor)
    create_friendship_tables(infra.db, infra.db_cursor)
    create_follow_request_table(infra.db, infra.db_cursor)
    create_question_tables(infra.db, infra.db_cursor)
    create_notification_table(infra.db, infra.db_cursor)
    # create_chat_tables(infra.db, infra.db_cursor)

    simulation_task = asyncio.create_task(infra.running())
    if inference_configs["model_type"][:3] == "gpt":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType(inference_configs["model_type"]),
        )

    try:
        all_topic_df = pd.read_csv("data/twitter_dataset/all_topics.csv")
        if "False" in csv_path or "True" in csv_path:
            if "-" not in csv_path:
                topic_name = csv_path.split("/")[-1].split(".")[0]
            else:
                topic_name = csv_path.split("/")[-1].split(".")[0].split(
                    "-")[0]
            source_post_time = (
                all_topic_df[all_topic_df["topic_name"] ==
                             topic_name]["start_time"].item().split(" ")[1])
            start_hour = int(source_post_time.split(":")[0]) + float(
                int(source_post_time.split(":")[1]) / 60)
    except Exception:
        social_log.info("No real-world data, let start_hour be 1PM")
        start_hour = 13

    agent_graph = await generate_agents(agent_info_path=csv_path,
                                        channel=channel,
                                        start_time=start_time,
                                        model=model,
                                        recsys_type=recsys_type,
                                        available_actions=available_actions,
                                        twitter=infra)
    # agent_graph.visualize("initial_social_graph.png")

    for timestep in range(1, num_timesteps + 1):
        clock.time_step = timestep * 3
        db_file = db_path.split("/")[-1]
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)

        # Custom twitter-style recsys with connection degree filtering
        await update_rec_table_filtered(
            infra,
            max_connection_degree=3,
            current_time=clock.time_step,
        )

        # 0.05 * timestep here means 3 minutes / timestep
        simulation_time_hour = start_hour + 0.05 * timestep
        tasks = []
        for node_id, agent in agent_graph.get_agents():
            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()
                threshold = agent.user_info.profile["other_info"][
                    "active_threshold"][int(simulation_time_hour % 24)]
                if agent_ac_prob < threshold:
                    tasks.append(agent.perform_action_by_llm())
            else:
                await agent.perform_action_by_hci()

        await asyncio.gather(*tasks)
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

        asyncio.run(
            running(**data_params,
                    **simulation_params,
                    inference_configs=inference_configs))
    else:
        asyncio.run(running())
    social_log.info("Simulation finished.")
