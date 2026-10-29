# Slurm commands (copy and paste)

Run from `/project/community/aiosman`. Most commands use `$JID`, the job ID.
Set it once in your terminal, then paste the commands as they are:

```bash
JID=162909          # the job ID that sbatch printed (or see squeue below)
```

---

## 1. See my jobs

```bash
squeue -u $USER                    # all my jobs now (ST: R = running, PD = waiting)
squeue -j $JID                     # one job
squeue -u $USER --start            # when waiting jobs should start
make watch                         # same as squeue -u $USER, wider columns
```

## 2. Read the log of a job

Training logs are `logs/orient_unet_<JID>`, inference logs are `logs/orient_eval_<JID>`.
`.out` holds the normal output, `.err` the errors and warnings.

```bash
tail -n 30 logs/*_$JID.out         # last 30 lines
tail -f logs/*_$JID.out            # live; Ctrl+C stops watching (the job keeps running)
tail -n 30 logs/*_$JID.err         # errors, if the job failed
grep "val " logs/orient_unet_$JID.out   # training: only the validation line of each epoch
```

## 3. Details of one job (also after it finished)

```bash
sacct -j $JID -X --format=JobID,JobName%24,Partition,State,Elapsed,Timelimit,ExitCode
scontrol show job $JID             # everything Slurm knows (only while queued/running)
seff $JID                          # after it ends: CPU and memory it really used
srun --jobid=$JID --overlap nvidia-smi   # while running: is the GPU busy?
```

## 4. My recent jobs

```bash
sacct -u $USER -S now-2days -X --format=JobID,JobName%24,Partition,State,Start,Elapsed,ExitCode
```
Change `2days` to `7days` for a week.

## 5. Submit jobs

```bash
make train-gpu CONFIG=blind_both_seed1        # training (general, 12 h)
make infer CONFIG=blind_both_seed1            # inference on the 28 test sets
make infer CONFIG=blind_both_seed1 AFTER=$JID # inference that waits until training job $JID ends OK
```
The plain `sbatch` form, if you need it:
```bash
sbatch --partition=general --qos=general_qos --time=12:00:00 --job-name=NAME script.sh
sbatch --dependency=afterok:$JID script.sh    # start only after job $JID finished without error
```

## 6. Change or stop a job

```bash
scancel $JID                                      # stop one job (cannot be undone)
scontrol update JobId=$JID TimeLimit=12:00:00     # change the time limit (see note)
scontrol hold $JID                                # keep a waiting job from starting
scontrol release $JID                             # let it start again
```
Note: you can only lower a running job's time limit. A waiting job can be set to any
value up to the partition's maximum (the fix for `PartitionTimeLimit`).

## 7. The cluster

```bash
sinfo -p general,preempt           # time limits and node states (idle / mix / alloc / down)
```

| Partition | QoS to use | Max time | Notes |
|---|---|---|---|
| `general` | `general_qos` | 12 h | not interrupted; what we use |
| `preempt` | `preempt_qos` | 2 days | can be stopped by others' jobs (`--requeue` restarts it) |
| `debug` | | 1 h | short tests |

The partition and the QoS must match, or sbatch refuses the job.

---

## What the codes mean

**Job state** (`ST` in squeue, `State` in sacct)

| Code | Meaning |
|---|---|
| `PD` PENDING | waiting in the queue |
| `R` RUNNING | running |
| `CG` COMPLETING | finishing |
| `CD` COMPLETED | finished OK (exit code 0:0) |
| `F` FAILED | ended with an error: read the `.err` log |
| `TO` TIMEOUT | hit its time limit |
| `CA` CANCELLED | stopped with scancel |
| `PR` PREEMPTED | stopped to make room for another job (preempt partition) |

**Why a job waits** (`NODELIST(REASON)` in squeue)

| Reason | Meaning |
|---|---|
| `Resources` | no free GPU yet; it starts when one frees up |
| `Priority` | other jobs are ahead in the queue |
| `Dependency` | waiting for the job given in `--dependency` |
| `DependencyNeverSatisfied` | that job failed, so this one will never start: `scancel` it |
| `PartitionTimeLimit` | asked for more time than the partition allows: lower it (section 6) |
| `QOSMaxWallDurationPerJobLimit` | same idea, but the QoS limit |
