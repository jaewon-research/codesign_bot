-- Friend request table for managing pending connection requests
CREATE TABLE IF NOT EXISTS friend_request (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    requester_id INTEGER NOT NULL,
    requestee_id INTEGER NOT NULL,
    requester_choice TEXT CHECK(requester_choice IN ('friend', 'close_friend')),
    requestee_choice TEXT CHECK(requestee_choice IN ('friend', 'close_friend')),
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (requester_id) REFERENCES user(user_id),
    FOREIGN KEY (requestee_id) REFERENCES user(user_id),
    UNIQUE(requester_id, requestee_id)
);

CREATE INDEX IF NOT EXISTS idx_friend_request_requester ON friend_request(requester_id);
CREATE INDEX IF NOT EXISTS idx_friend_request_requestee ON friend_request(requestee_id);
CREATE INDEX IF NOT EXISTS idx_friend_request_status ON friend_request(status);