-- General notification table
CREATE TABLE IF NOT EXISTS notification (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient_id INTEGER NOT NULL,  -- User receiving the notification
    sender_id INTEGER,  -- User who triggered the notification (can be NULL for system notifications)
    
    notification_type TEXT NOT NULL CHECK(notification_type IN (
        -- Friend/Connection
        'friend_request',
        'friend_accepted',
        'follow_request',
        
        -- Content Interactions - Likes
        'like_reply',
        'like_comment',
        'like_response',
        'like_note',
        
        -- Content Interactions - Reactions
        'emoji_reaction',
        
        -- Content Interactions - Comments
        'comment',
        'reply',
        'mention',
        
        -- Questions
        'response_request',
        'question_share',
        'question_response',
        'daily_question',
        
        -- Pings
        'ping',
        
        -- System
        'system'
    )),
    
    content_text TEXT NOT NULL,
    emoji TEXT,  -- For reaction notifications
    related_id INTEGER,  -- ID of related entity (question_id, response_id, post_id, etc.)
    related_type TEXT,  -- Type of related entity ('response', 'note', 'comment', etc.)
    redirect_url TEXT,  -- Where to go when clicking notification
    
    is_read BOOLEAN DEFAULT 0,
    is_visible BOOLEAN DEFAULT 1,  -- Can be hidden without deleting
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- For batching updates
    
    FOREIGN KEY (recipient_id) REFERENCES user(user_id),
    FOREIGN KEY (sender_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_notification_recipient ON notification(recipient_id);
CREATE INDEX IF NOT EXISTS idx_notification_sender ON notification(sender_id);
CREATE INDEX IF NOT EXISTS idx_notification_type ON notification(notification_type);
CREATE INDEX IF NOT EXISTS idx_notification_read ON notification(is_read);
CREATE INDEX IF NOT EXISTS idx_notification_visible ON notification(is_visible);
CREATE INDEX IF NOT EXISTS idx_notification_updated ON notification(updated_at);