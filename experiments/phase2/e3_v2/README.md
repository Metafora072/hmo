# E3-v2 execution path

This package is the preregistered replacement for the legacy full-prompt E3
prototype. P0-B establishes only the context/query execution contract:

1. serialize and tokenize the exact full prompt once;
2. prefill the memory context;
3. intervene on attention KV while recurrent state remains unchanged;
4. process the query suffix with logical positions from the original sequence;
5. expose answer logits only after the query suffix;
6. score or generate from a fresh, single-use post-intervention state.

There is intentionally no formal GPU runner yet. P0-C and P0-D must complete
the recurrent-state gate, fixed-byte interventions, alpha isolation, and run
manifest before E3-v2 experiments are allowed to execute.
