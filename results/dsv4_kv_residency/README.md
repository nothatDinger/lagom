# DSv4 KV residency experiment outputs

`entrypoint.sh` writes `server.log`, `client.log`, `analysis.log`, `raw_records.json`,
`unaccepted_tokens.csv`, `kv_hit_ratio_vs_confidence_gap.png`, and `report.md` here.
Generated large/runtime artifacts are intentionally not committed.
Set `RESULTS_DIR` to relocate the base `results` directory; the runner creates a
`dsv4_kv_residency` child directory beneath it.
