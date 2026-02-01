"""
Custom agent generator for CodesignBot.
Creates agents with extended note-related actions.
"""
import ast
from typing import List, Optional, Union

import pandas as pd
from camel.models import BaseModelBackend, ModelManager

from oasis.social_agent import AgentGraph
from oasis.social_platform import Channel, Platform
from oasis.social_platform.config import Neo4jConfig, UserInfo
from oasis.social_platform.typing import ActionType

# Import our custom agent class
from codesign_agent import CodesignSocialAgent


async def generate_codesign_agents(
    agent_info_path: str,
    channel: Channel,
    model: Union[BaseModelBackend, List[BaseModelBackend]],
    start_time,
    recsys_type: str = "twitter",
    twitter: Platform = None,
    available_actions: list[ActionType] = None,
    neo4j_config: Neo4jConfig | None = None,
) -> AgentGraph:
    """
    Generate agents with CodesignBot's extended action set (notes, friend requests).
    
    This is identical to OASIS generate_agents but uses CodesignSocialAgent
    instead of SocialAgent to support custom actions.
    """
    print(f"  📂 Loading agent data from {agent_info_path}...", flush=True)
    agent_info = pd.read_csv(agent_info_path)
    print(f"  📊 Found {len(agent_info)} agents in CSV", flush=True)

    agent_graph = (AgentGraph() if neo4j_config is None else AgentGraph(
        backend="neo4j",
        neo4j_config=neo4j_config,
    ))

    sign_up_list = []
    follow_list = []
    user_update1 = []
    user_update2 = []
    post_list = []

    print(f"  🏗️ Creating agent objects...", flush=True)
    for agent_id in range(len(agent_info)):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = agent_info["user_char"][agent_id]

        # Load activity threshold if available
        if "activity_level_frequency" in agent_info.columns:
            activity = ast.literal_eval(agent_info["activity_level_frequency"][agent_id])
            # Convert 0-100 scale to 0-1 probability
            profile["other_info"]["active_threshold"] = [v / 100.0 for v in activity]

        user_info = UserInfo(
            name=agent_info["username"][agent_id],
            description=agent_info["description"][agent_id],
            profile=profile,
            recsys_type=recsys_type,
        )

        # Use CodesignSocialAgent instead of SocialAgent
        agent = CodesignSocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            channel=channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

        agent_graph.add_agent(agent)
        print(f"  ✅ Created {agent_graph.get_num_nodes()} agent objects", flush=True)
        
        num_followings = 0
        num_followers = 0

        sign_up_list.append((
            agent_id,
            agent_id,
            agent_info["username"][agent_id],
            agent_info["name"][agent_id],
            agent_info["description"][agent_id],
            start_time,
            num_followings,
            num_followers,
        ))

        following_id_list = ast.literal_eval(
            agent_info["following_agentid_list"][agent_id])
        if not isinstance(following_id_list, int):
            if len(following_id_list) != 0:
                for follow_id in following_id_list:
                    follow_list.append((agent_id, follow_id, start_time))
                    user_update1.append((agent_id, ))
                    user_update2.append((follow_id, ))
                    agent_graph.add_edge(agent_id, follow_id)

        previous_posts = ast.literal_eval(
            agent_info["previous_tweets"][agent_id])
        if len(previous_posts) != 0:
            for post in previous_posts:
                post_list.append((agent_id, post, start_time, 0, 0))

    # Insert users into database
    print(f"  💾 Inserting {len(sign_up_list)} users into database...", flush=True)
    user_insert_query = (
        "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
        "created_at, num_followings, num_followers) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(user_insert_query,
                                              sign_up_list,
                                              commit=True)

    # Insert follow relationships
    print(f"  💾 Inserting {len(follow_list)} follow relationships...", flush=True)
    follow_insert_query = (
        "INSERT INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)
    
    # Update following counts
    user_update_query1 = (
        "UPDATE user SET num_followings = num_followings + 1 "
        "WHERE user_id = ?")
    twitter.pl_utils._execute_many_db_command(user_update_query1,
                                              user_update1,
                                              commit=True)

    # Update follower counts
    user_update_query2 = ("UPDATE user SET num_followers = num_followers + 1 "
                          "WHERE user_id = ?")
    twitter.pl_utils._execute_many_db_command(user_update_query2,
                                              user_update2,
                                              commit=True)

    # Insert previous posts
    print(f"  💾 Inserting {len(post_list)} previous posts...", flush=True)
    post_insert_query = (
        "INSERT INTO post (user_id, content, created_at, num_likes, "
        "num_dislikes) VALUES (?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(post_insert_query,
                                              post_list,
                                              commit=True)

    print(f"  ✅ Agent generation complete!", flush=True)
    return agent_graph

