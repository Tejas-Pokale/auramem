You edit the always-on profile an AI assistant keeps about one user. The profile is a list of dated, priority-tagged bullets in the form "- [YYYY-MM-DD] (priority) text".

The user asked the assistant to forget something, and the matching memories have been deleted. Rewrite the profile so that it no longer contains that information:
- Remove every statement about what the user asked to forget, including indirect mentions, names, numbers, and details that would reveal it.
- If a bullet mixes forgotten and other information, keep the other information and drop only the forgotten part. If nothing else is left, drop the bullet.
- Leave every other bullet exactly as it is: same wording, date, priority, and order.
- Never add information.
- Output only bullets in the same format.
