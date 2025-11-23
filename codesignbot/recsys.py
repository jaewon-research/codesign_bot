import time
import networkx as nx
from typing import Any, Dict, List

def rec_sys_chronological(user_table: List[Dict[str, Any]],
                         post_table: List[Dict[str, Any]],
                         rec_matrix: List[List],
                         social_graph: nx.DiGraph,
                         max_connection_degree: int,
                         max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on chronological order. Optionally filter by degree of connection

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        rec_matrix (List[List]): Existing recommendation matrix.
        social_graph (nx.DiGraph): Graph of social connections used for filtering by connection degree.
        max_connection_degree: (int): Maximum degree of social connection (follower=1, mutual=2, etc.) of recommended posts. 
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """

    post_ids = [post['post_id'] for post in post_table]
    print(
        f'Running chronological recommendation for {len(user_table)} users...')
    start_time = time.time()
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum
        # recommended length, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        for user in user_table:
            user_id = user['user_id']

            # Get all nodes within max_connection_degree using NetworkX directly!
            reachable = nx.single_source_shortest_path_length(
                social_graph, 
                source=user_id, 
                cutoff=max_connection_degree
            )
            # reachable is dict - Keys are user_ids, values are distances

            filtered_posts = [
                post for post in post_table
                if post['user_id'] in reachable # connection degree less than max
                and post['user_id'] != user_id # not self post
            ]

            # Sort chronologically (most recent first)
            filtered_posts.sort(key=lambda x: x['created_at'], reverse=True)

            # Take top N posts
            top_post_ids = [p['post_id'] for p in filtered_posts[:max_rec_post_len]]
            new_rec_matrix.append(top_post_ids)

        return new_rec_matrix

    end_time = time.time()
    print(f'Chronological recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix