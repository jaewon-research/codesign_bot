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

        # Compute date/recency scores for posts
        date_scores = np.array([
            np.log((271.8 - max(0, current_time - int(post.get('created_at', 0)))) / 100)
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
    post_table = fetch_table_from_db(platform.db_cursor, "post")
    rec_matrix = fetch_rec_table_as_matrix(platform.db_cursor)
    
    # Build social graph from current connections
    social_graph = build_social_graph(platform.db_cursor)
    
    # Run filtered recommendation
    new_rec_matrix = rec_sys_twhin_filtered(
        user_table=user_table,
        post_table=post_table,
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