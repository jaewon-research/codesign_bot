-- Participants in each chat room
CREATE TABLE IF NOT EXISTS chat_participant (
    room_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_read_message_id INTEGER,  -- Track what user has read
    PRIMARY KEY (room_id, user_id),
    FOREIGN KEY (room_id) REFERENCES chat_room(room_id),
    FOREIGN KEY (user_id) REFERENCES user(user_id),
    FOREIGN KEY (last_read_message_id) REFERENCES message(message_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_participant_user ON chat_participant(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_participant_room ON chat_participant(room_id);