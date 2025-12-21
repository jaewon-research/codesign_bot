-- Connection table for established friendships
-- Similar to WhoamI-Today backend, stores bidirectional relationships
CREATE TABLE IF NOT EXISTS connection (
    connection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user1_id INTEGER NOT NULL,
    user2_id INTEGER NOT NULL,
    user1_choice TEXT CHECK(user1_choice IN ('friend', 'close_friend')),
    user2_choice TEXT CHECK(user2_choice IN ('friend', 'close_friend')),
    user1_upgrade_time DATETIME,  -- When user1 upgraded to close_friend
    user2_upgrade_time DATETIME,  -- When user2 upgraded to close_friend
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user1_id) REFERENCES user(user_id),
    FOREIGN KEY (user2_id) REFERENCES user(user_id),
    UNIQUE(user1_id, user2_id),
    CHECK(user1_id < user2_id)  -- Ensure user1_id is always smaller to avoid duplicates
);

CREATE INDEX IF NOT EXISTS idx_connection_user1 ON connection(user1_id);
CREATE INDEX IF NOT EXISTS idx_connection_user2 ON connection(user2_id);
CREATE INDEX IF NOT EXISTS idx_connection_user1_choice ON connection(user1_id, user1_choice);
CREATE INDEX IF NOT EXISTS idx_connection_user2_choice ON connection(user2_id, user2_choice);