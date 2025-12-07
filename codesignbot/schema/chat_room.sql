-- Chat room table - supports both 1-on-1 DMs and group chats
CREATE TABLE IF NOT EXISTS chat_room (
    room_id INTEGER PRIMARY KEY AUTOINCREMENT,
    is_group BOOLEAN DEFAULT 0,  -- 0 = DM (2 people), 1 = group chat
    room_name TEXT,  -- NULL for DMs, name for groups
    active BOOLEAN DEFAULT 1,  -- Can be deactivated without deletion
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP  -- Last message time
);

CREATE INDEX IF NOT EXISTS idx_chat_room_active ON chat_room(active);
CREATE INDEX IF NOT EXISTS idx_chat_room_updated ON chat_room(updated_at);