-- Response table for daily question answers
CREATE TABLE IF NOT EXISTS question_response (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    daily_question_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    visibility TEXT DEFAULT 'friends' CHECK(visibility IN ('friends', 'close_friends')),
    is_edited BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(user_id),
    FOREIGN KEY (daily_question_id) REFERENCES daily_question(daily_question_id),
    UNIQUE(user_id, daily_question_id)  -- One response per user per question
);

CREATE INDEX IF NOT EXISTS idx_response_user ON question_response(user_id);
CREATE INDEX IF NOT EXISTS idx_response_question ON question_response(daily_question_id);
CREATE INDEX IF NOT EXISTS idx_response_visibility ON question_response(visibility);
CREATE INDEX IF NOT EXISTS idx_response_created ON question_response(created_at);

-- Track who has read each response
CREATE TABLE IF NOT EXISTS response_reader (
    response_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    read_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (response_id, user_id),
    FOREIGN KEY (response_id) REFERENCES question_response(response_id),
    FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_response_reader_user ON response_reader(user_id);