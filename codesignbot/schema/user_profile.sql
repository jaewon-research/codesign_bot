-- User profile table for codesignbot simulation
CREATE TABLE IF NOT EXISTS user_profile (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    username TEXT NOT NULL,
    email TEXT,
    bio TEXT,
    interests TEXT DEFAULT '[]', -- JSON array of interests e.g. ["#technology", "#science", "#art"]
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_profile_user_id ON user_profile(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profile_username ON user_profile(username);