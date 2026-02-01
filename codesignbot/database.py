import os
import sqlite3
import json
from datetime import date
import networkx as nx
from typing import List, Dict, Optional, Tuple

def get_schema_path() -> str:
    """Get the path to the codesignbot schema directory."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    schema_dir = os.path.join(curr_dir, 'schema')
    os.makedirs(schema_dir, exist_ok=True)
    return schema_dir


# ==================== Notes ========================

def create_note_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create the note, note_reader, note_like, and note_image tables."""
    schema_dir = get_schema_path()
    note_sql_path = os.path.join(schema_dir, 'note.sql')
    
    if os.path.exists(note_sql_path):
        with open(note_sql_path, 'r') as sql_file:
            cursor.executescript(sql_file.read())
        conn.commit()
        print("✓ Created note tables")
    else:
        print(f"⚠️  Schema file not found: {note_sql_path}")

def create_note(cursor: sqlite3.Cursor, user_id: int, content: str, 
                visibility: str = 'friends', created_at: int = None) -> int:
    """Create a new note.
    
    Args:
        cursor: Database cursor
        user_id: The user creating the note
        content: Note content
        visibility: 'friends' or 'close_friends'
        created_at: Optional simulation timestep (integer)
    
    Returns:
        note_id of the created note
    """
    if created_at is not None:
        cursor.execute("""
            INSERT INTO note (user_id, content, visibility, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, content, visibility, created_at))
    else:
        cursor.execute("""
            INSERT INTO note (user_id, content, visibility)
            VALUES (?, ?, ?)
        """, (user_id, content, visibility))
    return cursor.lastrowid


def like_note(cursor: sqlite3.Cursor, user_id: int, note_id: int) -> int:
    """Like a note.
    
    Args:
        cursor: Database cursor
        user_id: The user liking
        note_id: The note to like
    
    Returns:
        like_id of the created like
    """
    cursor.execute("""
        INSERT INTO note_like (user_id, note_id)
        VALUES (?, ?)
    """, (user_id, note_id))
    return cursor.lastrowid


def get_note_owner(cursor: sqlite3.Cursor, note_id: int) -> Optional[int]:
    """Get the owner of a note.
    
    Args:
        cursor: Database cursor
        note_id: The note ID
    
    Returns:
        user_id of the note owner, or None if not found
    """
    cursor.execute("SELECT user_id FROM note WHERE note_id = ?", (note_id,))
    row = cursor.fetchone()
    return row[0] if row else None


def create_note_comment(cursor: sqlite3.Cursor, user_id: int, note_id: int,
                        content: str, created_at: int = None) -> int:
    """Create a comment on a note.
    
    Args:
        cursor: Database cursor
        user_id: The user commenting
        note_id: The note to comment on
        content: Comment content
        created_at: Optional simulation timestep (integer)
    
    Returns:
        comment_id of the created comment
    """
    if created_at is not None:
        cursor.execute("""
            INSERT INTO note_comment (user_id, note_id, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, note_id, content, created_at))
    else:
        cursor.execute("""
            INSERT INTO note_comment (user_id, note_id, content, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (user_id, note_id, content))
    return cursor.lastrowid


def get_note_comments(cursor: sqlite3.Cursor, note_id: int) -> list:
    """Get all comments on a note.
    
    Args:
        cursor: Database cursor
        note_id: The note ID
    
    Returns:
        List of comment dictionaries
    """
    cursor.execute("""
        SELECT nc.comment_id, nc.user_id, nc.content, nc.created_at,
               u.user_name, u.name
        FROM note_comment nc
        JOIN user u ON nc.user_id = u.user_id
        WHERE nc.note_id = ?
        ORDER BY nc.created_at ASC
    """, (note_id,))
    
    comments = []
    for row in cursor.fetchall():
        comments.append({
            'comment_id': row[0],
            'user_id': row[1],
            'content': row[2],
            'created_at': row[3],
            'user_name': row[4],
            'name': row[5] or row[4]
        })
    return comments


# ==================== Comment Functions ====================

def create_comment(cursor: sqlite3.Cursor, user_id: int, post_id: int, 
                   content: str) -> int:
    """Create a comment on a post.
    
    Args:
        cursor: Database cursor
        user_id: The user commenting
        post_id: The post to comment on
        content: Comment content
    
    Returns:
        comment_id of the created comment
    """
    cursor.execute("""
        INSERT INTO comment (user_id, post_id, content, created_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, (user_id, post_id, content))
    return cursor.lastrowid


def like_comment(cursor: sqlite3.Cursor, user_id: int, comment_id: int) -> int:
    """Like a comment.
    
    Args:
        cursor: Database cursor
        user_id: The user liking
        comment_id: The comment to like
    
    Returns:
        like_id of the created like
    """
    cursor.execute("""
        INSERT INTO comment_like (user_id, comment_id)
        VALUES (?, ?)
    """, (user_id, comment_id))
    return cursor.lastrowid


def get_post_owner(cursor: sqlite3.Cursor, post_id: int) -> Optional[int]:
    """Get the owner of a post.
    
    Args:
        cursor: Database cursor
        post_id: The post ID
    
    Returns:
        user_id of the post owner, or None if not found
    """
    cursor.execute("SELECT user_id FROM post WHERE post_id = ?", (post_id,))
    row = cursor.fetchone()
    return row[0] if row else None


def get_comment_owner(cursor: sqlite3.Cursor, comment_id: int) -> Optional[int]:
    """Get the owner of a comment.
    
    Args:
        cursor: Database cursor
        comment_id: The comment ID
    
    Returns:
        user_id of the comment owner, or None if not found
    """
    cursor.execute("SELECT user_id FROM comment WHERE comment_id = ?", (comment_id,))
    row = cursor.fetchone()
    return row[0] if row else None

# ==================== User Profile ========================

def create_profile_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create the user_profile table in the database."""
    schema_dir = get_schema_path()
    profile_sql_path = os.path.join(schema_dir, 'user_profile.sql')
    
    if os.path.exists(profile_sql_path):
        with open(profile_sql_path, 'r') as sql_file:
            profile_sql_script = sql_file.read()
        cursor.executescript(profile_sql_script)
        conn.commit()
        print("✓ Created user_profile table")
    else:
        print(f"⚠️  Schema file not found: {profile_sql_path}")

def insert_user_profile(cursor: sqlite3.Cursor, user_id: int, 
                       username: str, email: str = None, 
                       bio: str = None) -> int:
    """Insert a new user profile."""
    cursor.execute("""
        INSERT INTO user_profile (user_id, username, email, bio)
        VALUES (?, ?, ?, ?)
    """, (user_id, username, email, bio))
    return cursor.lastrowid

def get_user_profile(cursor: sqlite3.Cursor, user_id: int) -> dict:
    """Get a user profile by user_id."""
    cursor.execute("""
        SELECT profile_id, user_id, username, email, bio, created_at, updated_at
        FROM user_profile
        WHERE user_id = ?
    """, (user_id,))
    
    row = cursor.fetchone()
    if row:
        return {
            'profile_id': row[0],
            'user_id': row[1],
            'username': row[2],
            'email': row[3],
            'bio': row[4],
            'created_at': row[5],
            'updated_at': row[6]
        }
    return None

def update_user_profile(cursor: sqlite3.Cursor, user_id: int, 
                       username: str = None, email: str = None, 
                       bio: str = None) -> None:
    """Update a user profile."""
    updates = []
    params = []
    
    if username is not None:
        updates.append("username = ?")
        params.append(username)
    if email is not None:
        updates.append("email = ?")
        params.append(email)
    if bio is not None:
        updates.append("bio = ?")
        params.append(bio)
    
    if updates:
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(user_id)
        
        query = f"""
            UPDATE user_profile
            SET {', '.join(updates)}
            WHERE user_id = ?
        """
        cursor.execute(query, params)

def create_friendship_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create the friend_request and connection tables."""
    schema_dir = get_schema_path()
    
    # Create friend_request table
    friend_request_sql_path = os.path.join(schema_dir, 'friend_request.sql')
    if os.path.exists(friend_request_sql_path):
        with open(friend_request_sql_path, 'r') as sql_file:
            cursor.executescript(sql_file.read())
        print("✓ Created friend_request table")
    
    # Create connection table
    connection_sql_path = os.path.join(schema_dir, 'connection.sql')
    if os.path.exists(connection_sql_path):
        with open(connection_sql_path, 'r') as sql_file:
            cursor.executescript(sql_file.read())
        print("✓ Created connection table")
    
    conn.commit()

# ==================== Friend Request ========================
# Friend Request Functions

def send_friend_request(cursor: sqlite3.Cursor, requester_id: int, 
                       requestee_id: int, requester_choice: str = 'friend') -> int:
    """Send a friend request."""
    cursor.execute("""
        INSERT INTO friend_request (requester_id, requestee_id, requester_choice, status)
        VALUES (?, ?, ?, 'pending')
    """, (requester_id, requestee_id, requester_choice))
    return cursor.lastrowid


def accept_friend_request(cursor: sqlite3.Cursor, request_id: int, 
                         requestee_choice: str = 'friend') -> bool:
    """Accept a friend request and create a connection."""
    # Get the request details
    cursor.execute("""
        SELECT requester_id, requestee_id, requester_choice
        FROM friend_request
        WHERE request_id = ? AND status = 'pending'
    """, (request_id,))
    
    row = cursor.fetchone()
    if not row:
        return False
    
    requester_id, requestee_id, requester_choice = row
    
    # Update the friend request
    cursor.execute("""
        UPDATE friend_request
        SET status = 'accepted', requestee_choice = ?, updated_at = CURRENT_TIMESTAMP
        WHERE request_id = ?
    """, (requestee_choice, request_id))
    
    # Create the connection (ensure user1_id < user2_id)
    user1_id, user2_id = (requester_id, requestee_id) if requester_id < requestee_id else (requestee_id, requester_id)
    user1_choice = requester_choice if requester_id < requestee_id else requestee_choice
    user2_choice = requestee_choice if requester_id < requestee_id else requester_choice
    
    cursor.execute("""
        INSERT INTO connection (user1_id, user2_id, user1_choice, user2_choice)
        VALUES (?, ?, ?, ?)
    """, (user1_id, user2_id, user1_choice, user2_choice))
    
    return True


def reject_friend_request(cursor: sqlite3.Cursor, request_id: int) -> bool:
    """Reject a friend request."""
    cursor.execute("""
        UPDATE friend_request
        SET status = 'rejected', updated_at = CURRENT_TIMESTAMP
        WHERE request_id = ? AND status = 'pending'
    """, (request_id,))
    return cursor.rowcount > 0


def get_pending_friend_requests(cursor: sqlite3.Cursor, user_id: int) -> list:
    """Get all pending friend requests for a user."""
    cursor.execute("""
        SELECT request_id, requester_id, requestee_id, requester_choice, created_at
        FROM friend_request
        WHERE requestee_id = ? AND status = 'pending'
        ORDER BY created_at DESC
    """, (user_id,))
    
    return [
        {
            'request_id': row[0],
            'requester_id': row[1],
            'requestee_id': row[2],
            'requester_choice': row[3],
            'created_at': row[4]
        }
        for row in cursor.fetchall()
    ]

# ==================== Connection ========================
# Connection Functions

def get_friends(cursor: sqlite3.Cursor, user_id: int) -> list:
    """Get all friends of a user (both friend and close_friend)."""
    cursor.execute("""
        SELECT 
            CASE 
                WHEN user1_id = ? THEN user2_id 
                ELSE user1_id 
            END as friend_id,
            CASE 
                WHEN user1_id = ? THEN user2_choice 
                ELSE user1_choice 
            END as friendship_level
        FROM connection
        WHERE user1_id = ? OR user2_id = ?
    """, (user_id, user_id, user_id, user_id))
    
    return [
        {
            'friend_id': row[0],
            'friendship_level': row[1]  # 'friend' or 'close_friend'
        }
        for row in cursor.fetchall()
    ]


def get_close_friends(cursor: sqlite3.Cursor, user_id: int) -> list:
    """Get only close friends of a user."""
    cursor.execute("""
        SELECT 
            CASE 
                WHEN user1_id = ? THEN user2_id 
                ELSE user1_id 
            END as friend_id
        FROM connection
        WHERE (user1_id = ? AND user2_choice = 'close_friend')
           OR (user2_id = ? AND user1_choice = 'close_friend')
    """, (user_id, user_id, user_id))
    
    return [row[0] for row in cursor.fetchall()]


def is_connected(cursor: sqlite3.Cursor, user1_id: int, user2_id: int) -> bool:
    """Check if two users are connected (friends)."""
    cursor.execute("""
        SELECT 1 FROM connection
        WHERE (user1_id = ? AND user2_id = ?)
           OR (user1_id = ? AND user2_id = ?)
        LIMIT 1
    """, (min(user1_id, user2_id), max(user1_id, user2_id), 
          min(user1_id, user2_id), max(user1_id, user2_id)))
    
    return cursor.fetchone() is not None


def is_close_friend(cursor: sqlite3.Cursor, user1_id: int, user2_id: int) -> bool:
    """Check if user2 is a close friend of user1."""
    cursor.execute("""
        SELECT 
            CASE 
                WHEN user1_id = ? THEN user2_choice 
                ELSE user1_choice 
            END as friendship_level
        FROM connection
        WHERE (user1_id = ? AND user2_id = ?)
           OR (user1_id = ? AND user2_id = ?)
    """, (user1_id, min(user1_id, user2_id), max(user1_id, user2_id),
          min(user1_id, user2_id), max(user1_id, user2_id)))
    
    row = cursor.fetchone()
    return row is not None and row[0] == 'close_friend'


def upgrade_to_close_friend(cursor: sqlite3.Cursor, user_id: int, friend_id: int) -> bool:
    """Upgrade a friendship to close friend."""
    user1_id, user2_id = (user_id, friend_id) if user_id < friend_id else (friend_id, user_id)
    
    if user_id == user1_id:
        cursor.execute("""
            UPDATE connection
            SET user1_choice = 'close_friend',
                user1_upgrade_time = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE user1_id = ? AND user2_id = ?
        """, (user1_id, user2_id))
    else:
        cursor.execute("""
            UPDATE connection
            SET user2_choice = 'close_friend',
                user2_upgrade_time = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE user1_id = ? AND user2_id = ?
        """, (user1_id, user2_id))
    
    return cursor.rowcount > 0


def remove_connection(cursor: sqlite3.Cursor, user1_id: int, user2_id: int) -> bool:
    """Remove a connection (unfriend)."""
    cursor.execute("""
        DELETE FROM connection
        WHERE (user1_id = ? AND user2_id = ?)
           OR (user1_id = ? AND user2_id = ?)
    """, (min(user1_id, user2_id), max(user1_id, user2_id),
          min(user1_id, user2_id), max(user1_id, user2_id)))
    
    return cursor.rowcount > 0


def get_mutual_friends(cursor: sqlite3.Cursor, user1_id: int, user2_id: int) -> list:
    """Get mutual friends between two users."""
    cursor.execute("""
        SELECT DISTINCT
            CASE 
                WHEN c1.user1_id = ? THEN c1.user2_id 
                ELSE c1.user1_id 
            END as mutual_friend_id
        FROM connection c1
        INNER JOIN connection c2 
            ON (
                (c1.user1_id = c2.user1_id AND c1.user2_id != c2.user2_id AND c2.user2_id = ?)
                OR (c1.user1_id = c2.user2_id AND c1.user2_id != c2.user1_id AND c2.user1_id = ?)
                OR (c1.user2_id = c2.user1_id AND c1.user1_id != c2.user2_id AND c2.user2_id = ?)
                OR (c1.user2_id = c2.user2_id AND c1.user1_id != c2.user1_id AND c2.user1_id = ?)
            )
        WHERE c1.user1_id = ? OR c1.user2_id = ?
    """, (user1_id, user2_id, user2_id, user2_id, user2_id, user1_id, user1_id))
    
    return [row[0] for row in cursor.fetchall()]

def set_interests(cursor: sqlite3.Cursor, user_id: int, interests: List[str]) -> bool:
    """
    Set the interests for a user profile (replaces existing interests).
    
    Args:
        cursor: Database cursor
        user_id: User ID
        interests: List of interest hashtags (e.g., ['#hiking', '#biking'])
    
    Returns:
        bool: True if successful, False if user not found
    """
    # Validate and format each interest
    validated_interests = []
    for interest in interests:
        # Ensure it starts with #
        if not interest.startswith('#'):
            interest = f"#{interest}"
        # Optional: convert to lowercase for consistency
        interest = interest.lower()
        # Remove duplicates
        if interest not in validated_interests:
            validated_interests.append(interest)
    
    # Convert to JSON string
    interests_json = json.dumps(validated_interests)
    
    # Update the profile
    cursor.execute("""
        UPDATE user_profile
        SET interests = ?, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
    """, (interests_json, user_id))
    
    return cursor.rowcount > 0

# ==================== Follow Request ========================
# Follow Request Functions

def create_follow_request_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create the follow_request table."""
    schema_dir = get_schema_path()
    follow_request_sql_path = os.path.join(schema_dir, 'follow_request.sql')
    
    if os.path.exists(follow_request_sql_path):
        with open(follow_request_sql_path, 'r') as sql_file:
            cursor.executescript(sql_file.read())
        print("✓ Created follow_request table")
        conn.commit()
    else:
        print(f"⚠️  Schema file not found: {follow_request_sql_path}")


def send_follow_request(cursor: sqlite3.Cursor, requester_id: int, requestee_id: int) -> int:
    """
    Send a follow request from requester to requestee.
    
    Returns:
        int: request_id if successful, or existing request_id if already exists
    """
    # Check if request already exists
    cursor.execute("""
        SELECT request_id, status
        FROM follow_request
        WHERE requester_id = ? AND requestee_id = ?
    """, (requester_id, requestee_id))
    
    existing = cursor.fetchone()
    if existing:
        # If pending, return existing request_id
        if existing[1] == 'pending':
            return existing[0]
        # If previously rejected, update to pending
        elif existing[1] == 'rejected':
            cursor.execute("""
                UPDATE follow_request
                SET status = 'pending', updated_at = CURRENT_TIMESTAMP
                WHERE request_id = ?
            """, (existing[0],))
            return existing[0]
    
    # Create new request
    cursor.execute("""
        INSERT INTO follow_request (requester_id, requestee_id, status)
        VALUES (?, ?, 'pending')
    """, (requester_id, requestee_id))
    return cursor.lastrowid


async def accept_follow_request(
    cursor: sqlite3.Cursor, 
    platform: 'Platform',  # Pass the Platform instance
    request_id: int
) -> dict:
    """
    Accept a follow request using the Platform's follow() method.
    
    Returns:
        dict: Result from platform.follow() with success status
    """
    # Get the request details
    cursor.execute("""
        SELECT requester_id, requestee_id
        FROM follow_request
        WHERE request_id = ? AND status = 'pending'
    """, (request_id,))
    
    row = cursor.fetchone()
    if not row:
        return {"success": False, "error": "Follow request not found or already processed"}
    
    requester_id, requestee_id = row
    
    # Use Platform's follow method (handles everything properly)
    result = await platform.follow(agent_id=requester_id, followee_id=requestee_id)
    
    if result.get("success"):
        # Update the follow request status
        cursor.execute("""
            UPDATE follow_request
            SET status = 'accepted', updated_at = CURRENT_TIMESTAMP
            WHERE request_id = ?
        """, (request_id,))
        platform.db.commit()
    
    return result


def reject_follow_request(cursor: sqlite3.Cursor, request_id: int) -> bool:
    """
    Reject a follow request.
    
    Returns:
        bool: True if successful
    """
    cursor.execute("""
        UPDATE follow_request
        SET status = 'rejected', updated_at = CURRENT_TIMESTAMP
        WHERE request_id = ? AND status = 'pending'
    """, (request_id,))
    
    return cursor.rowcount > 0


def cancel_follow_request(cursor: sqlite3.Cursor, requester_id: int, requestee_id: int) -> bool:
    """
    Cancel a pending follow request (requester cancels their own request).
    
    Returns:
        bool: True if successful
    """
    cursor.execute("""
        DELETE FROM follow_request
        WHERE requester_id = ? AND requestee_id = ? AND status = 'pending'
    """, (requester_id, requestee_id))
    
    return cursor.rowcount > 0


def get_pending_follow_requests(cursor: sqlite3.Cursor, user_id: int) -> list:
    """
    Get all pending follow requests for a user (requests they need to approve).
    
    Returns:
        list: List of dicts with request details
    """
    cursor.execute("""
        SELECT request_id, requester_id, created_at
        FROM follow_request
        WHERE requestee_id = ? AND status = 'pending'
        ORDER BY created_at DESC
    """, (user_id,))
    
    return [
        {
            'request_id': row[0],
            'requester_id': row[1],
            'created_at': row[2]
        }
        for row in cursor.fetchall()
    ]


def get_sent_follow_requests(cursor: sqlite3.Cursor, user_id: int) -> list:
    """
    Get all pending follow requests sent by a user.
    
    Returns:
        list: List of dicts with request details
    """
    cursor.execute("""
        SELECT request_id, requestee_id, created_at
        FROM follow_request
        WHERE requester_id = ? AND status = 'pending'
        ORDER BY created_at DESC
    """, (user_id,))
    
    return [
        {
            'request_id': row[0],
            'requestee_id': row[1],
            'created_at': row[2]
        }
        for row in cursor.fetchall()
    ]


def has_pending_follow_request(cursor: sqlite3.Cursor, requester_id: int, requestee_id: int) -> bool:
    """
    Check if there's a pending follow request from requester to requestee.
    
    Returns:
        bool: True if pending request exists
    """
    cursor.execute("""
        SELECT 1
        FROM follow_request
        WHERE requester_id = ? AND requestee_id = ? AND status = 'pending'
        LIMIT 1
    """, (requester_id, requestee_id))
    
    return cursor.fetchone() is not None


def get_follow_request_status(cursor: sqlite3.Cursor, requester_id: int, requestee_id: int) -> str:
    """
    Get the status of a follow request.
    
    Returns:
        str: 'pending', 'accepted', 'rejected', or None if no request exists
    """
    cursor.execute("""
        SELECT status
        FROM follow_request
        WHERE requester_id = ? AND requestee_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
    """, (requester_id, requestee_id))
    
    row = cursor.fetchone()
    return row[0] if row else None

# ==================== Social Graph ====================

def build_social_graph(cursor: sqlite3.Cursor) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from connection and follow tables.
    
    Edge weights represent connection "closeness":
    - close_friend: weight 0 (closest)
    - friend: weight 1
    - follow: weight 2
    """
    G = nx.DiGraph()
    
    # Add all users as nodes
    cursor.execute("SELECT user_id FROM user")
    for (user_id,) in cursor.fetchall():
        G.add_node(user_id)
    
    # Add connections (bidirectional friendships)
    cursor.execute("""
        SELECT user1_id, user2_id, user1_choice, user2_choice 
        FROM connection
    """)
    for user1_id, user2_id, user1_choice, user2_choice in cursor.fetchall():
        # Weight based on how the OTHER user categorized them
        weight1 = 0 if user2_choice == 'close_friend' else 1
        weight2 = 0 if user1_choice == 'close_friend' else 1
        G.add_edge(user1_id, user2_id, weight=weight1)
        G.add_edge(user2_id, user1_id, weight=weight2)
    
    # Add follow relationships (one-directional)
    cursor.execute("SELECT follower_id, followee_id FROM follow")
    for follower_id, followee_id in cursor.fetchall():
        if not G.has_edge(follower_id, followee_id):
            G.add_edge(follower_id, followee_id, weight=2)
    
    return G

# ==================== Notifications ====================

def create_notification(cursor: sqlite3.Cursor, recipient_id: int, 
                       notification_type: str, content_text: str,
                       sender_id: int = None, related_id: int = None) -> int:
    """Create a new notification."""
    cursor.execute("""
        INSERT INTO notification 
        (recipient_id, sender_id, notification_type, content_text, related_id)
        VALUES (?, ?, ?, ?, ?)
    """, (recipient_id, sender_id, notification_type, content_text, related_id))
    
    return cursor.lastrowid


def get_user_notifications(cursor: sqlite3.Cursor, user_id: int, 
                          unread_only: bool = False) -> List[Dict]:
    """Get notifications for a user."""
    query = """
        SELECT n.notification_id, n.sender_id, n.notification_type, 
               n.content_text, n.related_id, n.is_read, n.created_at,
               u.user_name as sender_name
        FROM notification n
        LEFT JOIN user u ON n.sender_id = u.user_id
        WHERE n.recipient_id = ?
    """
    
    if unread_only:
        query += " AND n.is_read = 0"
    
    query += " ORDER BY n.created_at DESC"
    
    cursor.execute(query, (user_id,))
    
    return [
        {
            'notification_id': row[0],
            'sender_id': row[1],
            'notification_type': row[2],
            'content_text': row[3],
            'related_id': row[4],
            'is_read': bool(row[5]),
            'created_at': row[6],
            'sender_name': row[7]
        }
        for row in cursor.fetchall()
    ]


def mark_notification_read(cursor: sqlite3.Cursor, notification_id: int) -> bool:
    """Mark a notification as read."""
    cursor.execute("""
        UPDATE notification
        SET is_read = 1
        WHERE notification_id = ?
    """, (notification_id,))
    
    return cursor.rowcount > 0


def mark_all_notifications_read(cursor: sqlite3.Cursor, user_id: int) -> int:
    """Mark all notifications for a user as read."""
    cursor.execute("""
        UPDATE notification
        SET is_read = 1
        WHERE recipient_id = ? AND is_read = 0
    """, (user_id,))
    
    return cursor.rowcount

# ==================== Daily Questions ====================

def create_question_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create all daily question related tables."""
    schema_dir = get_schema_path()
    
    tables = [
        'daily_question.sql',
        'question_response.sql',
        'notification.sql',
        'question_share.sql'
    ]
    
    for table_file in tables:
        table_path = os.path.join(schema_dir, table_file)
        if os.path.exists(table_path):
            with open(table_path, 'r') as sql_file:
                cursor.executescript(sql_file.read())
            print(f"✓ Created {table_file.replace('.sql', '')} table")
    
    conn.commit()

def get_daily_question(cursor: sqlite3.Cursor, target_date: date = None) -> Optional[Dict]:
    """Get the daily question for a specific date (defaults to today)."""
    if target_date is None:
        target_date = date.today()
    
    cursor.execute("""
        SELECT daily_question_id, question_text_en, question_text_ko, 
               scheduled_date, is_active
        FROM daily_question
        WHERE scheduled_date = ?
    """, (target_date,))
    
    row = cursor.fetchone()
    if row:
        return {
            'daily_question_id': row[0],
            'question_text_en': row[1],
            'question_text_ko': row[2],
            'scheduled_date': row[3],
            'is_active': bool(row[4])
        }
    return None


def set_active_question(cursor: sqlite3.Cursor, daily_question_id: int) -> bool:
    """Set a specific question as active (deactivates all others)."""
    # Deactivate all questions
    cursor.execute("UPDATE daily_question SET is_active = 0")
    
    # Activate the specified question
    cursor.execute("""
        UPDATE daily_question 
        SET is_active = 1, updated_at = CURRENT_TIMESTAMP
        WHERE daily_question_id = ?
    """, (daily_question_id,))
    
    return cursor.rowcount > 0


def get_active_question(cursor: sqlite3.Cursor) -> Optional[Dict]:
    """Get the currently active daily question."""
    cursor.execute("""
        SELECT daily_question_id, question_text_en, question_text_ko, 
               scheduled_date
        FROM daily_question
        WHERE is_active = 1
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    if row:
        return {
            'daily_question_id': row[0],
            'question_text_en': row[1],
            'question_text_ko': row[2],
            'scheduled_date': row[3]
        }
    return None


def notify_question_response(cursor: sqlite3.Cursor, responder_id: int, 
                            daily_question_id: int, response_text: str) -> List[int]:
    """
    Notify all users who shared this question with the responder.
    
    Returns:
        List[int]: List of notification_ids created
    """
    # Find all users who shared this question with the responder
    cursor.execute("""
        SELECT sender_id, share_id
        FROM question_share
        WHERE recipient_id = ? AND daily_question_id = ?
    """, (responder_id, daily_question_id))
    
    sharers = cursor.fetchall()
    notification_ids = []
    
    # Get responder name
    cursor.execute("SELECT user_name FROM user WHERE user_id = ?", (responder_id,))
    responder_name = cursor.fetchone()[0]
    
    # Get question text
    cursor.execute("""
        SELECT question_text_en, question_text_ko
        FROM daily_question
        WHERE daily_question_id = ?
    """, (daily_question_id,))
    question = cursor.fetchone()
    
    # Notify each sharer
    for sender_id, share_id in sharers:
        content = f"{responder_name} responded to the question you shared: {question[0]}"
        
        notification_id = create_notification(
            cursor,
            recipient_id=sender_id,
            sender_id=responder_id,
            notification_type='question_response',
            content_text=content,
            related_id=share_id  # Link to the share
        )
        notification_ids.append(notification_id)
    
    return notification_ids


# Update the submit_question_response function:
def submit_question_response(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                            user_id: int, daily_question_id: int, 
                            response_text: str, visibility: str = 'friends') -> int:
    """Submit a response to a daily question and notify sharers."""
    # Insert the response
    cursor.execute("""
        INSERT OR REPLACE INTO question_response 
        (user_id, daily_question_id, response_text, visibility)
        VALUES (?, ?, ?, ?)
    """, (user_id, daily_question_id, response_text, visibility))
    
    response_id = cursor.lastrowid
    
    # Notify users who shared this question with you
    notify_question_response(cursor, user_id, daily_question_id, response_text)
    
    conn.commit()
    return response_id


def get_user_response(cursor: sqlite3.Cursor, user_id: int, 
                     daily_question_id: int) -> Optional[Dict]:
    """Get a user's response to a specific daily question."""
    cursor.execute("""
        SELECT response_id, response_text, visibility, created_at
        FROM question_response
        WHERE user_id = ? AND daily_question_id = ?
    """, (user_id, daily_question_id))
    
    row = cursor.fetchone()
    if row:
        return {
            'response_id': row[0],
            'response_text': row[1],
            'visibility': row[2],
            'created_at': row[3]
        }
    return None


def get_friends_responses(cursor: sqlite3.Cursor, user_id: int, 
                         daily_question_id: int) -> List[Dict]:
    """Get responses from user's friends for a specific question."""
    cursor.execute("""
        SELECT qr.response_id, qr.user_id, qr.response_text, 
               qr.visibility, qr.created_at, u.user_name
        FROM question_response qr
        JOIN user u ON qr.user_id = u.user_id
        WHERE qr.daily_question_id = ?
        AND (
            qr.visibility = 'public'
            OR (qr.visibility = 'friends' AND qr.user_id IN (
                SELECT CASE 
                    WHEN user1_id = ? THEN user2_id 
                    ELSE user1_id 
                END
                FROM connection
                WHERE user1_id = ? OR user2_id = ?
            ))
            OR (qr.visibility = 'close_friends' AND qr.user_id IN (
                SELECT CASE 
                    WHEN user1_id = ? THEN user2_id 
                    ELSE user1_id 
                END
                FROM connection
                WHERE (user1_id = ? AND user2_choice = 'close_friend')
                   OR (user2_id = ? AND user1_choice = 'close_friend')
            ))
        )
        ORDER BY qr.created_at DESC
    """, (daily_question_id, user_id, user_id, user_id, user_id, user_id, user_id))
    
    return [
        {
            'response_id': row[0],
            'user_id': row[1],
            'response_text': row[2],
            'visibility': row[3],
            'created_at': row[4],
            'username': row[5]
        }
        for row in cursor.fetchall()
    ]


def share_question_with_friend(cursor: sqlite3.Cursor, sender_id: int, 
                               recipient_id: int, daily_question_id: int) -> Tuple[int, int]:
    """
    Share a daily question with a friend (creates notification).
    
    Returns:
        Tuple[int, int]: (share_id, notification_id)
    """
    # Get the question details
    cursor.execute("""
        SELECT question_text_en, question_text_ko
        FROM daily_question
        WHERE daily_question_id = ?
    """, (daily_question_id,))
    
    question = cursor.fetchone()
    if not question:
        raise ValueError("Question not found")
    
    # Get sender name
    cursor.execute("SELECT user_name FROM user WHERE user_id = ?", (sender_id,))
    sender_name = cursor.fetchone()[0]
    
    # Create notification
    content = f"{sender_name} shared a daily question with you: {question[0]}"
    notification_id = create_notification(
        cursor,
        recipient_id=recipient_id,
        sender_id=sender_id,
        notification_type='question_share',
        content_text=content,
        related_id=daily_question_id
    )
    
    # Record the share
    cursor.execute("""
        INSERT OR IGNORE INTO question_share 
        (sender_id, recipient_id, daily_question_id, notification_id)
        VALUES (?, ?, ?, ?)
    """, (sender_id, recipient_id, daily_question_id, notification_id))
    
    share_id = cursor.lastrowid
    return (share_id, notification_id)


def get_shared_questions(cursor: sqlite3.Cursor, user_id: int) -> List[Dict]:
    """Get all questions shared with a user."""
    cursor.execute("""
        SELECT qs.share_id, qs.sender_id, qs.daily_question_id,
               dq.question_text_en, dq.question_text_ko,
               u.user_name as sender_name, qs.created_at
        FROM question_share qs
        JOIN daily_question dq ON qs.daily_question_id = dq.daily_question_id
        JOIN user u ON qs.sender_id = u.user_id
        WHERE qs.recipient_id = ?
        ORDER BY qs.created_at DESC
    """, (user_id,))
    
    return [
        {
            'share_id': row[0],
            'sender_id': row[1],
            'daily_question_id': row[2],
            'question_text_en': row[3],
            'question_text_ko': row[4],
            'sender_name': row[5],
            'created_at': row[6]
        }
        for row in cursor.fetchall()
    ]

# ==================== Chat Rooms ====================

def create_chat_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create all chat/DM related tables."""
    schema_dir = get_schema_path()
    
    tables = [
        'chat_room.sql',
        'chat_participant.sql',
        'message.sql',
    ]
    
    for table_file in tables:
        table_path = os.path.join(schema_dir, table_file)
        if os.path.exists(table_path):
            with open(table_path, 'r') as sql_file:
                cursor.executescript(sql_file.read())
            print(f"✓ Created {table_file.replace('.sql', '')} table")
    
    conn.commit()

def get_or_create_dm_room(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                         user1_id: int, user2_id: int) -> int:
    """
    Get or create a DM room between two users.
    
    Returns:
        int: room_id
    """
    # Check if DM room already exists between these users
    cursor.execute("""
        SELECT cp1.room_id
        FROM chat_participant cp1
        JOIN chat_participant cp2 ON cp1.room_id = cp2.room_id
        JOIN chat_room cr ON cp1.room_id = cr.room_id
        WHERE cp1.user_id = ? AND cp2.user_id = ?
        AND cr.is_group = 0
        LIMIT 1
    """, (user1_id, user2_id))
    
    result = cursor.fetchone()
    if result:
        return result[0]
    
    # Create new DM room
    cursor.execute("""
        INSERT INTO chat_room (is_group, active)
        VALUES (0, 1)
    """)
    room_id = cursor.lastrowid
    
    # Add both users as participants
    cursor.executemany("""
        INSERT INTO chat_participant (room_id, user_id)
        VALUES (?, ?)
    """, [(room_id, user1_id), (room_id, user2_id)])
    
    conn.commit()
    return room_id


def create_group_chat(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                     creator_id: int, user_ids: List[int], 
                     room_name: str) -> int:
    """
    Create a group chat with multiple users.
    
    Returns:
        int: room_id
    """
    # Create group room
    cursor.execute("""
        INSERT INTO chat_room (is_group, room_name, active)
        VALUES (1, ?, 1)
    """, (room_name,))
    room_id = cursor.lastrowid
    
    # Add creator and all users as participants
    all_user_ids = list(set([creator_id] + user_ids))  # Deduplicate
    cursor.executemany("""
        INSERT INTO chat_participant (room_id, user_id)
        VALUES (?, ?)
    """, [(room_id, uid) for uid in all_user_ids])
    
    conn.commit()
    return room_id


def add_user_to_room(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                    room_id: int, user_id: int) -> bool:
    """Add a user to an existing chat room."""
    try:
        cursor.execute("""
            INSERT INTO chat_participant (room_id, user_id)
            VALUES (?, ?)
        """, (room_id, user_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # User already in room


def get_user_chat_rooms(cursor: sqlite3.Cursor, user_id: int) -> List[Dict]:
    """Get all chat rooms for a user with last message info."""
    cursor.execute("""
        SELECT cr.room_id, cr.is_group, cr.room_name, cr.active,
               cr.updated_at, m.content as last_message,
               m.sent_at as last_message_time, m.sender_id
        FROM chat_room cr
        JOIN chat_participant cp ON cr.room_id = cp.room_id
        LEFT JOIN message m ON cr.room_id = m.room_id
            AND m.message_id = (
                SELECT message_id 
                FROM message 
                WHERE room_id = cr.room_id 
                ORDER BY sent_at DESC 
                LIMIT 1
            )
        WHERE cp.user_id = ? AND cr.active = 1
        ORDER BY cr.updated_at DESC
    """, (user_id,))
    
    rooms = []
    for row in cursor.fetchall():
        room = {
            'room_id': row[0],
            'is_group': bool(row[1]),
            'room_name': row[2],
            'active': bool(row[3]),
            'updated_at': row[4],
            'last_message': row[5],
            'last_message_time': row[6],
            'last_sender_id': row[7]
        }
        
        # Get other participants (for DMs, get the other person's name)
        if not room['is_group']:
            cursor.execute("""
                SELECT u.user_id, u.user_name
                FROM chat_participant cp
                JOIN user u ON cp.user_id = u.user_id
                WHERE cp.room_id = ? AND cp.user_id != ?
            """, (row[0], user_id))
            other_user = cursor.fetchone()
            if other_user:
                room['other_user_id'] = other_user[0]
                room['other_user_name'] = other_user[1]
        
        rooms.append(room)
    
    return rooms

# ==================== Notifications ========================

def create_notification_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create the notification table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notification (
            notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipient_id INTEGER NOT NULL,
            sender_id INTEGER,
            notification_type TEXT NOT NULL CHECK(notification_type IN (
                'friend_request', 'friend_accepted', 'follow_request',
                'like_reply', 'like_comment', 'like_response', 'like_note',
                'emoji_reaction', 'comment', 'reply', 'mention',
                'response_request', 'question_share', 'question_response', 'daily_question',
                'ping', 'system'
            )),
            content_text TEXT NOT NULL,
            emoji TEXT,
            related_id INTEGER,
            related_type TEXT,
            redirect_url TEXT,
            is_read BOOLEAN DEFAULT 0,
            is_visible BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (recipient_id) REFERENCES user(user_id),
            FOREIGN KEY (sender_id) REFERENCES user(user_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_recipient ON notification(recipient_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_sender ON notification(sender_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_type ON notification(notification_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_read ON notification(is_read)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_visible ON notification(is_visible)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_updated ON notification(updated_at)")
    conn.commit()


# ==================== Message Management ====================

# def send_message(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
#                 room_id: int, sender_id: int, content: str,
#                 parent_id: int = None) -> int:
#     """
#     Send a message in a chat room.
    
#     Returns:
#         int: message_id
#     """
#     # Insert message
#     cursor.execute("""
#         INSERT INTO message (room_id, sender_id, content, parent_id)
#         VALUES (?, ?, ?, ?)
#     """, (room_id, sender_id, content, parent_id))
    
#     message_id = cursor.lastrowid
    
#     # Update room's last activity time
#     cursor.execute("""
#         UPDATE chat_room
#         SET updated_at = CURRENT_TIMESTAMP
#         WHERE room_id = ?
#     """, (room_id,))
    
#     conn.commit()
#     return message_id


# def get_room_messages(cursor: sqlite3.Cursor, room_id: int, 
#                      limit: int = 50, offset: int = 0) -> List[Dict]:
#     """Get messages from a chat room (paginated)."""
#     cursor.execute("""
#         SELECT m.message_id, m.sender_id, m.content, m.parent_id, m.sent_at,
#                u.user_name as sender_name
#         FROM message m
#         JOIN user u ON m.sender_id = u.user_id
#         WHERE m.room_id = ?
#         ORDER BY m.sent_at DESC
#         LIMIT ? OFFSET ?
#     """, (room_id, limit, offset))
    
#     messages = []
#     for row in cursor.fetchall():
#         messages.append({
#             'message_id': row[0],
#             'sender_id': row[1],
#             'content': row[2],
#             'parent_id': row[3],
#             'sent_at': row[4],
#             'sender_name': row[5]
#         })
    
#     return list(reversed(messages))  # Return in chronological order


# def mark_messages_read(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
#                       room_id: int, user_id: int) -> bool:
#     """Mark all messages in a room as read by updating last_read_message_id."""
#     # Get the latest message in the room
#     cursor.execute("""
#         SELECT message_id
#         FROM message
#         WHERE room_id = ?
#         ORDER BY sent_at DESC
#         LIMIT 1
#     """, (room_id,))
    
#     result = cursor.fetchone()
#     if not result:
#         return False
    
#     last_message_id = result[0]
    
#     # Update participant's last read message
#     cursor.execute("""
#         UPDATE chat_participant
#         SET last_read_message_id = ?
#         WHERE room_id = ? AND user_id = ?
#     """, (last_message_id, room_id, user_id))
    
#     conn.commit()
#     return cursor.rowcount > 0


# def get_unread_count(cursor: sqlite3.Cursor, room_id: int, user_id: int) -> int:
#     """Get unread message count for a user in a room."""
#     cursor.execute("""
#         SELECT cp.last_read_message_id
#         FROM chat_participant cp
#         WHERE cp.room_id = ? AND cp.user_id = ?
#     """, (room_id, user_id))
    
#     result = cursor.fetchone()
#     last_read_id = result[0] if result and result[0] else 0
    
#     # Count messages after last read
#     cursor.execute("""
#         SELECT COUNT(*)
#         FROM message
#         WHERE room_id = ? AND message_id > ? AND sender_id != ?
#     """, (room_id, last_read_id, user_id))
    
#     return cursor.fetchone()[0]


# def get_total_unread_count(cursor: sqlite3.Cursor, user_id: int) -> int:
#     """Get total unread message count across all rooms."""
#     cursor.execute("""
#         SELECT SUM(
#             (SELECT COUNT(*) 
#              FROM message m 
#              WHERE m.room_id = cp.room_id 
#              AND m.message_id > COALESCE(cp.last_read_message_id, 0)
#              AND m.sender_id != cp.user_id)
#         )
#         FROM chat_participant cp
#         WHERE cp.user_id = ?
#     """, (user_id,))
    
#     result = cursor.fetchone()[0]
#     return result if result else 0


# ==================== Simulation Metadata ====================

def create_simulation_meta_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create the simulation_meta table to store simulation timing configuration."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulation_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            start_time INTEGER NOT NULL,
            simulation_hours REAL NOT NULL DEFAULT 24,
            num_timesteps INTEGER NOT NULL,
            seconds_per_timestep REAL NOT NULL,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
    """)
    conn.commit()
    print("✓ Created simulation_meta table")


def set_simulation_meta(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                        start_time: int, simulation_hours: float, 
                        num_timesteps: int) -> dict:
    """Set simulation metadata. Replaces any existing metadata.
    
    Args:
        cursor: Database cursor
        conn: Database connection
        start_time: Unix timestamp when simulation starts
        simulation_hours: Total duration of simulation in hours (e.g., 24)
        num_timesteps: Number of timesteps in the simulation
    
    Returns:
        dict with the stored metadata including calculated seconds_per_timestep
    """
    seconds_per_timestep = (simulation_hours * 3600) / num_timesteps
    
    # Delete any existing metadata and insert new
    cursor.execute("DELETE FROM simulation_meta")
    cursor.execute("""
        INSERT INTO simulation_meta (id, start_time, simulation_hours, num_timesteps, seconds_per_timestep)
        VALUES (1, ?, ?, ?, ?)
    """, (start_time, simulation_hours, num_timesteps, seconds_per_timestep))
    conn.commit()
    
    return {
        "start_time": start_time,
        "simulation_hours": simulation_hours,
        "num_timesteps": num_timesteps,
        "seconds_per_timestep": seconds_per_timestep
    }


def get_simulation_meta(cursor: sqlite3.Cursor) -> Optional[dict]:
    """Get simulation metadata.
    
    Returns:
        dict with start_time, simulation_hours, num_timesteps, seconds_per_timestep
        or None if no metadata exists
    """
    cursor.execute("""
        SELECT start_time, simulation_hours, num_timesteps, seconds_per_timestep
        FROM simulation_meta WHERE id = 1
    """)
    row = cursor.fetchone()
    if row:
        return {
            "start_time": row[0],
            "simulation_hours": row[1],
            "num_timesteps": row[2],
            "seconds_per_timestep": row[3]
        }
    return None


def timestep_to_unix(cursor: sqlite3.Cursor, timestep: int) -> Optional[int]:
    """Convert a timestep number to Unix timestamp.
    
    Args:
        cursor: Database cursor
        timestep: The timestep number (0, 1, 2, ...)
    
    Returns:
        Unix timestamp for that timestep, or None if no metadata
    """
    meta = get_simulation_meta(cursor)
    if meta is None:
        return None
    return int(meta["start_time"] + (timestep * meta["seconds_per_timestep"]))


def unix_to_timestep(cursor: sqlite3.Cursor, unix_time: int) -> Optional[int]:
    """Convert a Unix timestamp to the nearest timestep number.
    
    Args:
        cursor: Database cursor
        unix_time: Unix timestamp
    
    Returns:
        Timestep number, or None if no metadata
    """
    meta = get_simulation_meta(cursor)
    if meta is None:
        return None
    if meta["seconds_per_timestep"] == 0:
        return 0
    return int((unix_time - meta["start_time"]) / meta["seconds_per_timestep"])
