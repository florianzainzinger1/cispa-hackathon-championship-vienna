# Live Training Monitor - Quick Commands

## Einfacher Status Check
```bash
cd /p/home/jusers/zainzinger1/jureca/HackathonVienna/task_2_text_attribution

# Einmaliger Check
./scripts/monitor_v3.sh

# Auto-refresh alle 10 Sekunden
watch -n 10 ./scripts/monitor_v3.sh
```

## Jobs Checken
```bash
# Job Status in der Queue
squeue -u zainzinger1

# Spezifische Jobs (ModernBERT: 14304171, DeBERTa: 14304172)
squeue -j 14304171,14304172
```

## Logs Live Verfolgen
```bash
# ModernBERT Logs
tail -f logs/log_modernbert_large_v3_14304171.err

# DeBERTa Logs
tail -f logs/log_deberta_large_v3_14304172.err

# Beide parallel (in zwei Terminals)
tail -f logs/log_*_v3_143041*.err
```

##  Fortschritt Extrahieren
```bash
# Letzte F1 Scores
grep "eval_f1_macro" logs/log_*_v3_143041*.err | tail -5

# Training Loss
grep "'loss':" logs/log_*_v3_143041*.err | tail -10

# Aktuelle Epoche
grep "'epoch':" logs/log_*_v3_143041*.err | tail -2
```

## Modelle Checken (wenn fertig)
```bash
# Check ob Modelle gespeichert wurden
ls -lh /p/scratch/training2562/zainzinger1/model_*_v3_trained/

# Model Config ansehen
cat /p/scratch/training2562/zainzinger1/model_modernbert_large_v3_trained/config.json
```
