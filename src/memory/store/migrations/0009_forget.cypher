// Forget (Phase 8): a preview waits for the user's confirmation before anything is deleted,
// and every deletion leaves a content-free audit event.

CREATE CONSTRAINT pending_forget_key IF NOT EXISTS
FOR (p:PendingForget) REQUIRE p.preview_id IS UNIQUE;

CREATE INDEX pending_forget_user IF NOT EXISTS
FOR (p:PendingForget) ON (p.user_id, p.session_id);

CREATE INDEX deletion_event_user IF NOT EXISTS
FOR (d:DeletionEvent) ON (d.user_id);
