-- Note table for general posts/status updates
-- Similar to OASIS post but with WhoamI visibility controls
CREATE TABLE IF NOT EXISTS note (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    visibility TEXT DEFAULT 'friends' CHECK(visibility IN ('friends', 'close_friends')),
    is_edited BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_note_user ON note(user_id);
CREATE INDEX IF NOT EXISTS idx_note_visibility ON note(visibility);
CREATE INDEX IF NOT EXISTS idx_note_created ON note(created_at);

-- Track who has read each note
CREATE TABLE IF NOT EXISTS note_reader (
    note_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    read_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (note_id, user_id),
    FOREIGN KEY (note_id) REFERENCES note(note_id),
    FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_note_reader_user ON note_reader(user_id);

-- Track who has liked each note
CREATE TABLE IF NOT EXISTS note_like (
    like_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    note_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, note_id),
    FOREIGN KEY (user_id) REFERENCES user(user_id),
    FOREIGN KEY (note_id) REFERENCES note(note_id)
);

CREATE INDEX IF NOT EXISTS idx_note_like_note ON note_like(note_id);
CREATE INDEX IF NOT EXISTS idx_note_like_user ON note_like(user_id);

-- Comments on notes
CREATE TABLE IF NOT EXISTS note_comment (
    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (note_id) REFERENCES note(note_id),
    FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_note_comment_note ON note_comment(note_id);
CREATE INDEX IF NOT EXISTS idx_note_comment_user ON note_comment(user_id);

-- Optional: Support for images attached to notes
CREATE TABLE IF NOT EXISTS note_image (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    image_url TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (note_id) REFERENCES note(note_id)
);

CREATE INDEX IF NOT EXISTS idx_note_image_note ON note_image(note_id);