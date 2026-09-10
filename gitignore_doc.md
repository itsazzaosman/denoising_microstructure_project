# How to ignore files in this repo


## Pattern cheatsheet

| Pattern | Meaning |
|---|---|
| `datasets/` | the whole directory (preferred) |
| `logs/*` | contents of `logs/`, directory itself still visited |
| `*.h5` | that extension anywhere in the repo |
| `**/euler_maps/` | that directory name at any depth |
| `/build` | only `build` at the repo root |
| `!keep.txt` | exception — un-ignore this file. Put it *after* the broad rule, and it will **not** work if the parent was ignored as `dir/` (use `dir/*` + `!dir/keep.txt`) |

