# Dataset Inference Attack - Aktueller Stand

**Score: 0.200 (TPR @ FPR = 5%)**  
**Datum: 14.12.2024**

---

## Übersicht

Dieses Dokument beschreibt den Ansatz, der erstmals einen Score von **0.200** auf dem Leaderboard erreicht hat. Das ist eine Verbesserung von **+42%** gegenüber dem vorherigen besten Score (0.141).

### Aufgabe
- **Ziel:** Für 1000 Subsets (je 100 Bilder) bestimmen, welche im Training des Klassifikators verwendet wurden
- **Metrik:** TPR @ FPR = 5% (True Positive Rate bei 5% False Positive Rate)
- **Submission:** CSV mit `subset_id` (0-999) und `membership` Score (0.0-1.0)

---

## Model-Architektur

```
ResNet18 (modifiziert für 28x28 Grayscale-Bilder)
├── conv1: Conv2d(3, 64, kernel_size=5, stride=1, padding=3, bias=False)
├── ... (Standard ResNet18 Blöcke)
└── fc: Sequential(
        Dropout(p=0.2)    ← WICHTIG: Regularisierung!
        Linear(512, 9)     ← 9 Klassen
    )
```

**Wichtige Eigenschaften:**
- Input: 28x28 Pixel, 3 Kanäle (ursprünglich Grayscale, zu RGB konvertiert)
- 9 Klassen (0-8)
- **Dropout(0.2):** Das Model ist gut regularisiert, was das Membership-Signal schwächt

---

## Der Erfolgreiche Ansatz: Extremwert-Methode

### Kernidee

> **Standard-Ansätze** berechnen Durchschnittswerte (Mean NLL, Mean Confidence).  
> **Unser Ansatz** fokussiert auf **Extremwerte** - die "schlechtesten" Samples pro Subset.

**Hypothese:** Wenn ein Subset im Training war, sollte das Model auch auf den schwierigsten Bildern besser performen. Member-Subsets haben weniger extreme Ausreißer.

### Die 3 Features

| Feature | Berechnung | Richtung |
|---------|------------|----------|
| `nll_max` | `nll.max()` - Höchster NLL im Subset | Niedriger = Member |
| `margin_min` | `margin.min()` - Niedrigster Margin | Höher = Member |
| `conf_p10` | `confidence.quantile(0.1)` - 10% Perzentil | Höher = Member |

### Feature-Berechnung im Detail

```python
for subset in subsets:
    images = subset['images']  # (100, 3, 28, 28)
    labels = subset['labels']  # (100,)
    
    # Forward Pass
    logits = model(images)     # (100, 9)
    probs = F.softmax(logits, dim=1)
    
    # 1. NLL (Negative Log-Likelihood)
    true_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    nll = -torch.log(true_probs.clamp(min=1e-10))
    nll_max = nll.max().item()  ← EXTREMWERT
    
    # 2. Confidence (Max Probability)
    conf = probs.max(dim=1).values
    conf_p10 = torch.quantile(conf, 0.1).item()  ← 10% PERZENTIL
    
    # 3. Margin (True Class Logit - Best Other Logit)
    true_logit = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    # Berechne max_other (höchster Logit außer true class)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), False)
    other_logits = logits.clone()
    other_logits[~mask] = float('-inf')
    max_other = other_logits.max(dim=1).values
    margin = true_logit - max_other
    margin_min = margin.min().item()  ← EXTREMWERT
```

### Score-Kombination via Rank-Average

```python
ranks = np.zeros(1000)

# Feature 1: nll_max (niedriger = besser = Member)
ranks += np.argsort(np.argsort(nll_max_values))

# Feature 2: margin_min (höher = besser = Member)  
ranks += np.argsort(np.argsort(-margin_min_values))

# Feature 3: conf_p10 (höher = besser = Member)
ranks += np.argsort(np.argsort(-conf_p10_values))

# Finaler Score: niedrigerer Rank = höherer Score
membership_scores = 1 - (ranks / 3 / 999)
```

**Warum Rank-Average statt direkte Werte?**
- Robust gegenüber Ausreißern
- Keine Annahmen über Feature-Verteilungen
- Gleichmäßige Gewichtung aller Features

---

## Warum funktioniert Extremwert besser als Durchschnitt?

| Aspekt | Durchschnitt (Mean) | Extremwert (Max/Min) |
|--------|---------------------|----------------------|
| **Problem** | Outlier werden geglättet | Erfasst genau die Outlier |
| **Signal** | Schwächer bei regularisierten Models | Stärker bei Extremfällen |
| **Robustheit** | Kann durch wenige gute Samples verfälscht werden | Fokussiert auf schwierigste Fälle |

---

## Fehlgeschlagene Ansätze (zum Vergleich)

Diese Methoden haben **NICHT** zu einer Verbesserung über 0.200 geführt:

| Methode | Beschreibung | Grund für Misserfolg |
|---------|--------------|----------------------|
| GMM Kalibrierung | 2-Komponenten GMM auf Scores | Zu wenig Trennung |
| MC Dropout | Unsicherheit via mehrere Forward Passes | Zusätzliche Varianz hilft nicht |
| Z-Score statt Rank | Standardisierung statt Ranking | Anfälliger für Ausreißer |
| Geometrisches Mittel | Produkt statt Summe | Zu empfindlich auf kleine Werte |
| Augmentation Consistency | Stabilität unter Transformationen | Korreliert mit bestehenden Features |
| Per-Class Analysis | Features pro Klasse | Erhöht Varianz ohne Nutzen |

---

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `best_attack.py` | Saubere Implementierung des besten Ansatzes |
| `submission_best.csv` | Beste Submission (Score: 0.200) |
| `ext_combo_extreme.csv` | Original-Submission die 0.200 erreichte |

---

## Nutzung

```bash
# Submission generieren
python best_attack.py

# Mit direktem Submit
python best_attack.py --submit
```

---

## Statistiken der Feature-Verteilungen

### nll_max (Höchster NLL pro Subset)
- Range: [1.11, 5.65]
- Mean: 2.34
- Std: 0.65

### margin_min (Niedrigster Margin pro Subset)
- Range: [-2.5, 1.8]
- Mean: 0.12
- Std: 0.48

### conf_p10 (10% Perzentil Confidence)
- Range: [0.19, 0.47]
- Mean: 0.35
- Std: 0.05

---

## Mögliche Verbesserungen

1. **Shadow Models:** Eigene Models trainieren und Unterschiede analysieren
2. **Gradient-basierte Features:** Wie stark reagieren Gradients auf Member vs Non-Member?
3. **Weitere Extremwerte:** worst-1, worst-2, worst-5 NLL systematisch testen
4. **Ensemble:** Mehrere Scoring-Methoden kombinieren

---

## Fazit

Der Extremwert-Ansatz erreicht **0.200 TPR @ FPR 5%**, was bedeutet:
- Bei 5% False Positives erkennen wir 20% der echten Member
- Das ist 4x besser als Zufall (5%)
- +42% Verbesserung gegenüber dem vorherigen Ansatz

Die Schlüsselerkenntnis: **Fokus auf Worst-Case statt Average-Case**.
