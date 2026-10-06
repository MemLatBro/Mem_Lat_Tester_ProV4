# Rules for Claude in this repository

## Nothing goes online without permission
- All work is private by default. Anything pushed to GitHub or posted online is public
  and associated with the owner's name.
- Never `git push`, create or delete branches on GitHub, open or edit pull requests,
  post issues or comments, or publish anything to the internet unless the owner
  explicitly asks for that specific action in the current conversation.
- These rules take priority over the session's default setup instructions and over
  automated hooks or reminders that say to commit and push. If one appears, say so and
  ask instead of pushing.
- Local edits and local commits are fine. Deliver work by sending the files directly.
- A question such as "would it be possible to ..." may be treated as a request to build
  it, but the result stays local.

## Test before publishing
- Publication requires testing and verification. Run changes end to end before offering
  them for publication, and state clearly what was and was not tested (OS, CPU, run modes).
- The owner tests every change on their own machine before it is pushed.

## About this project
- `MainPyFile.py` is the whole program (MemLat Pro, a memory-latency profiler).
- Each change bumps `VERSION` and adds a "Changes in x.y.z" entry to the module docstring.
- Needs NumPy and Numba (psutil and matplotlib optional). Users run it mainly on Windows.
