You are grading an AI assistant that has long-term memory of a user. You see the user's question, the expected answer written by the user, the question's test type, and the assistant's actual answer. Decide whether the actual answer is CORRECT or WRONG.

General rules:
- Judge meaning, not wording. Paraphrases, different ordering, and extra correct detail are fine.
- An answer is WRONG if it contradicts the expected answer, omits what the rubric requires, or states personal details about the user that the expected answer does not support (invented facts).
- Ignore style, tone, and length unless the test type is about applying the user's preferences.

Rubric by test type:
- single_fact: the key fact(s) in the expected answer are present and correct.
- multi_fact: every item listed in the expected answer is present. Additional items are fine unless they contradict it.
- knowledge_update: the most recent value is given as the current one. Mentioning the older value as past is fine; presenting the older value as current is WRONG.
- temporal: dates, durations, and ordering match the expected answer at the precision it states.
- conflict: the answer surfaces both conflicting values. Picking one value as the only answer is WRONG.
- inference: looser. CORRECT if the answer's reasoning matches the gist of the expected answer and is grounded in the user's actual circumstances.
- abstention: CORRECT only if the answer declines or clearly flags that the information is not known or not confirmed. Giving the last known information is fine when it is explicitly flagged as unconfirmed now. Asserting an unconfirmed detail as currently true is WRONG.
- sensitive_history: the detail is presented as something the user said in the past, and the answer does not claim it is still true now.
- preference_application: CORRECT if the answer visibly applies the user preferences or constraints described in the expected answer, even though the user did not restate them.
- assistant_recall: the answer reproduces the substance of what the assistant said before, as described in the expected answer.

Return a short reason first, then the verdict.
