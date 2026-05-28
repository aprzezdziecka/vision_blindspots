# Experiment Log — Vision Blind Spots

Plik do śledzenia parametrów użytych do wygenerowania każdego pliku CSV wyników.

---

## Struktura katalogów

```
results_A/   — Metoda A: osobny opis każdego obrazka → kategoryzacja różnic
results_B/   — Metoda B: bezpośrednie porównanie pary obrazków przez Qwena
```

---

## results_A/

### `descriptions_caltech_250_tokens.csv`

| Parametr | Wartość |
|---|---|
| Notebook | `method_A_strong_blindspots.ipynb` |
| Dataset | Caltech-101 |
| SELECTION | `["clip_caltech_blind_spots.csv", clip_imagenet_test_blind_spots.csv", 
 "dino_caltech_blind_spots.csv", "dino_imagenet_test_blind_spots.csv", 
 "siglip_caltech_blind_spots.csv", "siglip_imagenet_test_blind_spots.csv"]` |
| REF_THRESHOLD | 0.65 |
| QWEN_ID | `Qwen/Qwen2-VL-2B-Instruct` |
| MAX_NEW_TOKENS_DESC | 500 |
| min_pixels / max_pixels | 256×28×28 / 512×28×28 |

**DESCRIBE_PROMPT:**
```
Describe this image precisely. Focus on:
1. Main subject: exact species, breed, or type; color, size, distinctive markings
2. Subject's pose, orientation, and position in the frame
3. Background: specific objects present, scene type, environment details
4. Overall color palette and lighting conditions
Write 3-4 sentences in plain text.
```

**Uwagi:**
Najlepsze wyniki ze wszytskich uzyskiwanych prób pominięto siglip2 ze wzgledu na dużą liczbę zdjęć wyliczenia były dłuższe, a nie było pewności, że opisy zadziałają dobrze.

---

### `dino_imagenet_test_siglip_caltech_siglip_imagenet_test.csv`

| Parametr | Wartość |
|---|---|
| Notebook | `method_A_strong_blindspots.ipynb` |
| Dataset | Caltech-101 |
| SELECTION | `"caltech"` |
| REF_THRESHOLD | 0.65 |
| QWEN_ID | `Qwen/Qwen2-VL-2B-Instruct` |
| MAX_NEW_TOKENS_DESC | 250 |
| min_pixels / max_pixels | 256×28×28 / 512×28×28 |

**DESCRIBE_PROMPT:**
```
Describe this image precisely. Focus on:
1. Main subject: exact species, breed, or type; color, size, distinctive markings
2. Subject's pose, orientation, and position in the frame
3. Background: specific objects present, scene type, environment details
4. Overall color palette and lighting conditions
Write 3-4 sentences in plain text.
```

**Uwagi:**
Działało potencjalnie dobrze jedynie czasami urawne opisy, więc próba zwiekszenia liczby tokenów.

---

### `categorized_caltech.csv`

| Parametr | Wartość |
|---|---|
| Notebook | `method_A_strong_blindspots.ipynb` |
| Dataset | Caltech-101 |
| Wejście | `descriptions_caltech_250_tokens.csv` |
| QWEN_ID | `Qwen/Qwen2-VL-2B-Instruct` |
| MAX_NEW_TOKENS_CAT | 30 |
| Prompt wersja | v2 |

**CATEGORIZE_TEMPLATE (v2 — z definicjami inline):**
```
[Image 1] {desc_1}

[Image 2] {desc_2}

Select ALL labels that apply:
SUBJECT_TYPE     — subjects are different species, breed, or object category
SUBJECT_APPEARANCE — same type but different color, markings, size, or texture
SUBJECT_POSE     — different pose, angle, or position in frame
BACKGROUND_OBJECTS — different objects visible in the background
BACKGROUND_SETTING — different location type (indoor/outdoor, urban/nature, etc.)
COLORS_LIGHTING  — different dominant colors or lighting conditions
NO_DIFFERENCE    — descriptions are essentially the same

Output: comma-separated labels only.
```

**Kategorie:**
`SUBJECT_TYPE`, `SUBJECT_APPEARANCE`, `SUBJECT_POSE`, `BACKGROUND_OBJECTS`, `BACKGROUND_SETTING`, `COLORS_LIGHTING`, `NO_DIFFERENCE`

**Uwagi:**
Od niższego podejścia różnicą było to, że jeden przypadek z caltecha dostał jako no diff a reszta podobne rozłożenie jak poniższy plik.

---

### `categorized_caltech_PROMPT_v1.csv`

| Parametr | Wartość |
|---|---|
| Notebook | `method_A_strong_blindspots.ipynb` |
| Dataset | Caltech-101 |
| Wejście | |
| QWEN_ID | `Qwen/Qwen2-VL-2B-Instruct` |
| MAX_NEW_TOKENS_CAT | 30 |
| Prompt wersja | v1 |

**CATEGORIZE_TEMPLATE (v1 — lista bez definicji):**
```
Compare these two image descriptions and identify differences.

Image 1: {desc_1}

Image 2: {desc_2}

From the list below, pick ALL categories where the images differ:
SUBJECT_TYPE, SUBJECT_APPEARANCE, SUBJECT_POSE,
BACKGROUND_OBJECTS, BACKGROUND_SETTING, COLORS_LIGHTING, NO_DIFFERENCE

Reply with only the matching category names, comma-separated. No explanation.
```

**Uwagi:**
Jeśli kategoria pojawiała się dla jakiegoś modelu to pozostałe kategorie też się pojawiały. Wyniki niepozwalające na wyciąganie wniosków o kategorii blind spotu modelu. Potencjalnie przypisywane losowo.

---

## results_B/

### `results_B_strong_imagenet_QWEN_PROMPT_v2.csv`

| Parametr | Wartość |
|---|---|
| Notebook | `method_B_strong_blindspots.ipynb` |
| Dataset | ImageNet-1k (test) |
| SELECTION | `"imagenet"` |
| REF_THRESHOLD | 0.65 |
| QWEN_ID | `Qwen/Qwen2-VL-2B-Instruct` |
| MAX_NEW_TOKENS | 150 |
| Prompt wersja | v2 |
| Filtr | score_ref_1 < 0.65 AND score_ref_2 < 0.65 |

**QWEN_PROMPT (v2 — agresywny, wymusza szukanie różnic):**
```
These two images come from a dataset of visual blind spots — pairs that look similar
but contain real, specific differences. Your job is to find them.

Examine both images pixel by pixel. Look at:
- background details (objects, textures, patterns)
- lighting and shadows
- colors and saturation
- object positions, sizes, and orientations
- small details: text, logos, accessories, decorations
- cropping and framing

You MUST describe what is different. Even subtle differences like slight color shift,
minor rotation, or a missing small element count.
Never say the images are identical — there is always something.
Write a detailed paragraph in plain text.
```

**Uwagi:**
Nie działał najgorzej, ale czasami w momencie gdy zdjęcia przedstawiały zupełnie co inengo zamiast skupić się na tym, że na jednym jest niedżiwedź a na prawym czarno białe zdjęcie chłopca to opisywał, że nasycenie i pozycja niedźwiedzia i osoby jest taka sama.

---

### `results_B_strong_caltech_QWEN_PROMPT_v2.csv`

| Parametr | Wartość |
|---|---|
| Notebook | `method_B_strong_blindspots.ipynb` |
| Dataset | Caltech-101 |
| SELECTION | `"caltech"` |
| REF_THRESHOLD | |
| QWEN_ID | `Qwen/Qwen2-VL-2B-Instruct` |
| MAX_NEW_TOKENS | 150 |
| Prompt wersja | v2 |
| Filtr | score_ref_1 < REF_THRESHOLD AND score_ref_2 < REF_THRESHOLD |

**QWEN_PROMPT:** (identyczny jak wyżej — v2)

**Uwagi:**
Podobnie jak wyżej opisuje wszystko tylko nie to co rzeczywiście się różni.

---

### `results_B_clip_caltech_QWEN_PROMPT_v3.csv`

| Parametr | Wartość |
|---|---|
| Notebook | `method_B_strong_blindspots.ipynb` |
| Dataset | Caltech-101 |
| SELECTION | |
| REF_THRESHOLD | |
| QWEN_ID | `Qwen/Qwen2-VL-2B-Instruct` |
| MAX_NEW_TOKENS | 5 |
| Prompt wersja | v3 |
| Filtr | |

**QWEN_PROMPT (v3 — tak/nie o tle):**
```
Look at these two images. Do the backgrounds differ between the two images?
Answer with a single word: yes or no.
```

**Uwagi:**
Równy podział i brak możliwości weryfikacji co zdecydowało o takim przypisaniu (w przypadku opisów widzimy, gdy opisuje coś inaczej niż jest rzeczywiście i można ocenić potencjalną wartość takiego dopasowania).

---

## Prompty — pełna lista wersji

| Wersja | Plik | Podejście | MAX_NEW_TOKENS |
|---|---|---|---|
| v1 | method_B | Ogólne porównanie pary, lista różnic | 150 |
| v2 | method_B | Agresywny, wymusza znajdowanie różnic | 150 |
| v3 | method_B | Tak/nie — czy tła się różnią | 5 |
| v4 | method_B | Strukturyzowany: SUBJECT + BACKGROUND osobno | 150 |
| desc (A) | method_A | Opis pojedynczego obrazka | 120–250 |
| cat v1 (A) | method_A | Kategoryzacja z listy (bez definicji) | 30 |
| cat v2 (A) | method_A | Kategoryzacja z definicjami inline | 30 |

---

