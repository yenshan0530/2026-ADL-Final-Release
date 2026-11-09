# Run viewer

`docs/visualizer.html` shows what happened in a run. Open it in a browser (no server needed;
the files are read on your computer and never uploaded) and either drop a folder on it or
use *Choose folder…*:

- a run folder, `results/run_economy_<time>/` or `results/run_market_<time>/`: every
  market of the run, plus the run's standings and statistics;
- one market folder, `results/<run>/markets/<market id>/`;
- or single files: a market's `final.json`, with `transcript.jsonl` and `agents.jsonl`.

For the market you pick, it shows:

- **Scores:** each agent's profit (sellers) or surplus (buyers) in that market, ranked within its
  role as in the standings;
- **Stock:** which seller holds how many units of each item (coloured by kind), and how many
  it sold;
- **Who talked to whom:** sellers on the left, buyers on the right; a line's width grows
  with the messages and offers between two agents. Click a line or an agent to read only
  that conversation;
- **The conversation, round by round:** every message, room post, offer, acceptance,
  board purchase and review, with rejected actions marked and each agent's private note
  after its turn. Filter by a pair, an agent or a room, or search the text;
- **The run:** number of markets, turns, trades and messages, the share of the possible
  gains from trade realized, and the student standings.

A market folder needs `final.json`; `transcript.jsonl` (always written by the harness) adds
every action and private note. With `final.json` alone the viewer shows the messages and
trades.
