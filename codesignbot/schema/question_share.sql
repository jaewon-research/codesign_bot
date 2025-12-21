-- Track when users share daily questions with friends
CREATE TABLE IF NOT EXISTS question_share (
    share_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER NOT NULL,
    recipient_id INTEGER NOT NULL,
    daily_question_id INTEGER NOT NULL,
    notification_id INTEGER,  -- Link to the notification created
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sender_id) REFERENCES user(user_id),
    FOREIGN KEY (recipient_id) REFERENCES user(user_id),
    FOREIGN KEY (daily_question_id) REFERENCES daily_question(daily_question_id),
    FOREIGN KEY (notification_id) REFERENCES notification(notification_id),
    UNIQUE(sender_id, recipient_id, daily_question_id)  -- Can't share same question twice to same person
);

CREATE INDEX IF NOT EXISTS idx_question_share_sender ON question_share(sender_id);
CREATE INDEX IF NOT EXISTS idx_question_share_recipient ON question_share(recipient_id);
CREATE INDEX IF NOT EXISTS idx_question_share_question ON question_share(daily_question_id);