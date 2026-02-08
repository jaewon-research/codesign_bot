-- Follow request table for managing follow requests
-- Similar to Instagram's follow request system (unidirectional)
CREATE TABLE IF NOT EXISTS follow_request (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    requester_id INTEGER NOT NULL,  -- User who wants to follow
    requestee_id INTEGER NOT NULL,  -- User being followed
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (requester_id) REFERENCES user(user_id),
    FOREIGN KEY (requestee_id) REFERENCES user(user_id),
    UNIQUE(requester_id, requestee_id)
);

CREATE INDEX IF NOT EXISTS idx_follow_request_requester ON follow_request(requester_id);
CREATE INDEX IF NOT EXISTS idx_follow_request_requestee ON follow_request(requestee_id);
CREATE INDEX IF NOT EXISTS idx_follow_request_status ON follow_request(status);