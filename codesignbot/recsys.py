import time
import networkx as nx
import numpy as np
import torch
from typing import Any, Dict, List
from sklearn.metrics.pairwise import cosine_similarity

from oasis.social_platform.platform import Platform
from oasis.social_platform.database import fetch_table_from_db, fetch_rec_table_as_matrix

# Import twitter recsys from oasis module
from oasis.social_platform.recsys import get_recsys_model
from oasis.social_platform.process_recsys_posts import generate_post_vector

from database import build_social_graph


def parse_created_at(created_at_value, current_time: int = 0) -> int:
    """
    Parse created_at to Unix timestamp.
    
    Handles:
    - Unix timestamps (integers): 1706800000, 1706828800, etc.
    - Integer strings: "1706800000"
    - Datetime strings (legacy): '2026-01-12 00:36:24' (treated as current time)
    
    For datetime strings, treats them as "current" (most recent).
    """
    if created_at_value is None:
        return current_time  # Treat missing as current
    if isinstance(created_at_value, int):
        return created_at_value
    if isinstance(created_at_value, str):
        try:
            # Try parsing as integer string first
            return int(created_at_value)
        except ValueError:
            # It's a datetime string - treat as current time (most recent)
            return current_time
    return current_time


def compute_recency_score(post_created_at: int, current_time: int, max_age_hours: float = 24.0) -> float:
    """
    Compute a recency score for a post.
    
    Args:
        post_created_at: Unix timestamp when post was created
        current_time: Current Unix timestamp
        max_age_hours: Maximum age to consider (posts older than this get minimum score)
    
    Returns:
        Score between 0.1 (oldest) and 1.0 (newest)
    """
    age_seconds = max(0, current_time - post_created_at)
    max_age_seconds = max_age_hours * 3600
    
    # Normalize to 0-1 range, with newer posts getting higher scores
    # Posts older than max_age get a minimum score of 0.1
    if age_seconds >= max_age_seconds:
        return 0.1
    
    # Linear decay from 1.0 (newest) to 0.1 (max_age)
    normalized_age = age_seconds / max_age_seconds
    return 1.0 - (0.9 * normalized_age)

# Global cache for twhin model
twhin_tokenizer = None
twhin_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rec_sys_twhin_filtered(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    social_graph: nx.DiGraph,
    max_connection_degree: int,
    max_rec_post_len: int,
    current_time: int = 0,
) -> List[List]:
    """
    Recommend posts using twhin-bert similarity, filtered by degree of connection.

    Args:
        user_table: List of users.
        post_table: List of posts.
        rec_matrix: Existing recommendation matrix.
        social_graph: Graph of social connections for filtering.
        max_connection_degree: Max degree of connection for posts to consider.
        max_rec_post_len: Maximum number of recommended posts.
        current_time: Current simulation time (for recency scoring).

    Returns:
        List[List]: Updated recommendation matrix.
    """
    global twhin_tokenizer, twhin_model

    # Load twhin model if not already loaded
    if twhin_tokenizer is None or twhin_model is None:
        twhin_tokenizer, twhin_model = get_recsys_model(recsys_type="twhin-bert")

    post_ids = [post['post_id'] for post in post_table]
    print(
        f'Running twhin-bert filtered recommendation for {len(user_table)} users...'
    )
    start_time = time.time()
    new_rec_matrix = []

    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum
        # recommended length, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # Build user profiles for embedding
        user_profiles = []
        for user in user_table:
            if user.get('bio', None):
                user_profiles.append(user['bio'])
            else:
                user_profiles.append('This user does not have a profile')

        # Build post content list
        post_contents = [post['content'] for post in post_table]

        # Generate embeddings for all users and posts at once (batch for efficiency)
        corpus = user_profiles + post_contents
        all_vectors = generate_post_vector(
            twhin_model, twhin_tokenizer, corpus, batch_size=1000
        )
        user_vectors = all_vectors[:len(user_profiles)]
        post_vectors = all_vectors[len(user_profiles):]

        # Compute date/recency scores for posts using Unix timestamps
        date_scores = np.array([
            compute_recency_score(
                parse_created_at(post.get('created_at'), current_time),
                current_time,
                max_age_hours=24.0  # Use simulation_hours here if available
            )
            for post in post_table
        ])

        # Compute cosine similarity matrix: users x posts
        similarity_matrix = cosine_similarity(user_vectors, post_vectors)

        # Apply recency weighting
        weighted_similarity = similarity_matrix * date_scores

        # FIlter and rank posts for each user
        for user_idx, user in enumerate(user_table):
            user_id = user['user_id']

            # Get all nodes within max_connection_degree using NetworkX directly!
            reachable = nx.single_source_shortest_path_length(
                social_graph, 
                source=user_id, 
                cutoff=max_connection_degree
            )
            # reachable is dict - Keys are user_ids, values are distances

            # Get indices of posts from reachable users (excluding self)
            filtered_indices = [
                i for i, post in enumerate(post_table)
                if post['user_id'] in reachable and post['user_id'] != user_id
            ]

            # Get similarity scores for filtered posts
            user_similarities = weighted_similarity[user_idx, filtered_indices]

            # Rank by similarity and take top N
            top_k = min(max_rec_post_len, len(filtered_indices))
            top_local_indices = np.argsort(user_similarities)[::-1][:top_k]

            # Map to post IDs
            top_post_ids = [post_table[filtered_indices[i]]['post_id'] for i in top_local_indices]

            new_rec_matrix.append(top_post_ids)

    end_time = time.time()
    print(f'Twhin-bert filtered recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix


async def update_rec_table_filtered(
    platform: Platform,
    max_connection_degree: int,
    current_time: int,
) -> None:
    """
    Update recommendation table using twhin-bert with connection filtering.
    """
    # Fetch tables
    user_table = fetch_table_from_db(platform.db_cursor, "user")
    # Fetch notes instead of posts
    platform.db_cursor.execute("""
        SELECT note_id as post_id, user_id, content, created_at
        FROM note
        ORDER BY created_at DESC
    """)
    columns = ['post_id', 'user_id', 'content', 'created_at']
    note_table = [dict(zip(columns, row)) for row in platform.db_cursor.fetchall()]
    
    rec_matrix = fetch_rec_table_as_matrix(platform.db_cursor)
    
    # Build social graph from current connections
    social_graph = build_social_graph(platform.db_cursor)
    
    # Run filtered recommendation
    new_rec_matrix = rec_sys_twhin_filtered(
        user_table=user_table,
        post_table=note_table, # Use notes instead of posts
        rec_matrix=rec_matrix,
        social_graph=social_graph,
        max_connection_degree=max_connection_degree,
        max_rec_post_len=platform.max_rec_post_len,
        current_time=current_time,
    )
    
    # Update rec table in database
    for user_index, rec_post_ids in enumerate(new_rec_matrix):
        user_id = user_index + 1  # Assuming user_id = index + 1
        # Clear old recommendations
        platform.db_cursor.execute("DELETE FROM rec WHERE user_id = ?", (user_id,))
        # Insert new recommendations
        for post_id in rec_post_ids:
            platform.db_cursor.execute(
                "INSERT INTO rec (user_id, post_id) VALUES (?, ?)",
                (user_id, post_id)
            )
    platform.db.commit()