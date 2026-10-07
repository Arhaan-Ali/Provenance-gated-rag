## registry.py — declaration flow (AFTER M3, not before)

1. NOW (1 line): add "origin_strategy": "folder" to the save_decisions payload
   — without it, a folder-based decisions file + git-author origins = silent
   total mismatch, everything falls to D, hours lost debugging.

2. Replace the per-origin loop with ONE multi-select pass:
   "which did you write?" → A, "which are your team's?" → B,
   "which outside sources do you trust?" → C, rest → D unasked.

3. Flat corpora (no subfolders): group by signal, never list files.
   "12 git-committed by you / 38 unknown / 150 download-flagged" = 3 rows, not 200.

4. Declaring a folder = everything inside INHERITS. Then challenge exceptions
   opt-out style ("8 of these look downloaded — keep / drop / show me").
   Never opt-in per file, that turns 1 decision into 50.

5. Run 2+ asks ONLY about what's new. Declared folders keep their tier.
   Otherwise people mash Enter every run and the gate becomes decoration.

RULE: a signal can push a tier DOWN or challenge a claim. It can NEVER
promote on its own. Down = safe, up = a guess.

Don't touch: ReviewDecision shape, tier_for_document, decisions file format.
New file when the time comes: signals.py (git author, quarantine xattr,
Zone.Identifier). registry.py stays untouched by it.