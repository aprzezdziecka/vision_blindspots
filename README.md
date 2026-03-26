# Analyzing "Blind Spots" in Vision Models
Goal: Identify "hard" image pairs that fool a specific model (Base Model) into seeing them as identical, while other models (Support Models) correctly identify them as distinct. You will then use Vision Language Models (VLMs) to explain these differences.
## Milestone 1
**Step 1: Environment setup**
```bash
uv init vision_blindspots
cd vision_blindspots
# adding libraries required for the project
uv add torch torchvision transformers datasets pillow numpy scikit-learn
# adding developer tools for checking format (ruff) and types (mypy, pyright, ty)
uv add --dev ruff mypy pyright ty
# synchronizing the virtual environment
uv sync
```

**Step 2: Methodology**  
Our methodology is based on the paper [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209). This paper identifies a key problem in how multimodal 
models understand images despite their advanced reasoning capabilities. The root of these shortcomings lies in the pre-trained CLIP (Contrastive Language-Image Pre-Training) vision encoders that most of these systems rely on.
The authors analysed that different 

CLIP (Contrastive Language-Image Pre-Training) vision encoders that most of these systems rely on. CLIP encoders tend to focus heavily on high-level semantic understanding while overlooking intricate visual details

DINOv2, a vision-only self-supervised learning (SSL) model that captures finer visual and structural details



3 modele, jeden bedzie bazowy, 2 kontrolne
3 zbiory danych: linki

pipeline do kazdego z modeli, ktory wypluje nam embedding dla zdjecia
faiss do usprawnienia porownan zdjec, opisac, on da nam np 5 najbardziej podobnych zdjec dla kazdego zdjecia dla bazowego modelu (usuniemy duplikaty)
model bazowy: teraz dla mozliwych kandydatow liczymy poodbienstwo cosinusowe, wybieramy te ktore przekracza prog (tez do ustalenia, na poczatek 0.95 na podstawie artykulu)
modele kontrolne: te wybrane pary sprawdzamy tez cosunoswo na kontrolnych i jesli co najmniej 1 nie przekrocza progu ( na poczatek 0.6) czyli nie zgodzi sie ze zdjecia sa podobne to to jest nasz blind spot (do dopytania czy one sie musza zgodzic czy nie)



