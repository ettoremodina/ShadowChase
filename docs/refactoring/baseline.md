# Pre-refactor characterization baseline

This suite protects behavior observed before the repository and logging refactor.
It intentionally tests public effects rather than implementation structure so files
can move later without silently changing game semantics.

## Run the suite

Activate the CUDA-enabled project environment and run:

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pytest
```

To exclude the hardware-specific test:

```powershell
python -m pytest -m "not cuda"
```

The suite covers:

- initial state, movement, tickets, double moves, reveals, and history;
- random-agent move contracts;
- current pickle and JSON save round trips;
- `ml_logger` manifests, metrics, summaries, and lifecycle;
- feature-vector and DQN topology stability on CPU and CUDA;
- the PettingZoo observation and action-mask surface.

## Known pre-existing issues

- The repository dependency manifest is not currently resolvable as written.
- Some older saved pickle files reference removed classes such as
  `ScotlandYardMovement`; recent saves remain readable.
- The legacy JSON game export includes a NetworkX graph inside win-condition
  parameters and currently fails standard JSON serialization.
- The default `ml_logger` lifecycle can retain a lock on `catalog.sqlite` on
  Windows. The desired release behavior is recorded as an expected failure.
- Importing DQN modules can emit a circular-import warning depending on import
  order.
- The PettingZoo adapter relies on compatibility dictionary attributes instead
  of overriding the newer `observation_space()` and `action_space()` methods.

These issues are baseline findings, not changes introduced by the test suite.
