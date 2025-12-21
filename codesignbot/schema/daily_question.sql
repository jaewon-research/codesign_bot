-- Daily question table - tracks which question is active on each day
CREATE TABLE IF NOT EXISTS daily_question (
    daily_question_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text_en TEXT NOT NULL,
    question_text_ko TEXT NOT NULL,
    scheduled_date DATE NOT NULL UNIQUE,  -- One question per day
    is_active BOOLEAN DEFAULT 0,  -- Currently active question
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_daily_question_date ON daily_question(scheduled_date);
CREATE INDEX IF NOT EXISTS idx_daily_question_active ON daily_question(is_active);