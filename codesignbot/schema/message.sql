-- Messages in chat rooms
CREATE TABLE IF NOT EXISTS message (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id INTEGER NOT NULL,
    sender_id INTEGER NOT NULL,
    parent_id INTEGER,  -- For replies/threads
    content TEXT NOT NULL,
    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (room_id) REFERENCES chat_room(room_id),
    FOREIGN KEY (sender_id) REFERENCES user(user_id),
    FOREIGN KEY (parent_id) REFERENCES message(message_id)
);

CREATE INDEX IF NOT EXISTS idx_message_room ON message(room_id);
CREATE INDEX IF NOT EXISTS idx_message_sender ON message(sender_id);
CREATE INDEX IF NOT EXISTS idx_message_sent_at ON message(sent_at);
CREATE INDEX IF NOT EXISTS idx_message_parent ON message(parent_id);