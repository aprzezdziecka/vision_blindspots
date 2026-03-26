# Analyzing "Blind Spots" in Vision Models
Goal: Identify "hard" image pairs that fool a specific model (Base Model) into seeing them as identical, while other models (Support Models) correctly identify them as distinct. We will then use Vision Language Models (VLMs) to explain these differences.

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
models understand images despite their advanced reasoning capabilities. What might seem like a very easy question e.g. is the dog facing left or right from the camera's perspective can pose a big problem for a LLM. The root of these shortcomings lies in the pre-trained CLIP (Contrastive Language-Image Pre-Training) vision encoders that most of these systems rely on. The authors found CLIP-blind pairs (comparing ClIP embeddings with DINOv2). A CLIP-blind pair is characterized by having very similar CLIP embeddings but distinctly different DINOv2 embeddings.  
They indentified nine visual patterns that the CLIP vision encoders might consistently misinterpret:
- Orientation and Direction
- Presence of Specific Features
- State and Condition
- Quantity and Count
- Positional and Relational Context
- Color and Appearance
- Structural and Physical Characteristics
- Text
- Viewpoint and Perspective.
We hope to use those ideas in our project.

We will use three different models:
- openai/clip-vit-base-patch16 - CLIP (Contrastive Language-Image Pre-Training) vision encoders that most of the systems rely on. It is trained to align images with natural language descriptions. CLIP encoders tend to focus heavily on high-level semantic understanding while overlooking complex visual details. 
- google/siglip-base-patch16-224 - Google's refined version of CLIP.  It improves upon CLIP by using a sigmoid loss instead of a global softmax.
- facebook/dinov2-base - A Self-supervised Vision Transformer Model that captures finer visual and structural details. It focuses on pixel-level relationships and geometry.

The datasets we want to use: ILSVRC/imagenet-1k + pixparse/cc3m-wds + Caltech101

We will choose one model to be the base one e.g. CLIP, the other two will be refrence models.  

**The algorithm for finding blind spots:**
1. We will build a pipeline that will extract embeddings of images from all three models: base and reference.
2. To find the blind spots we will use cosine similarity.
3. Given the scale of the datasets, we can't count that similarity on every pair. To help with that we want to use Faiss (Facebook AI Similarity Search) - a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other.
   - metric: METRIC_INNER_PRODUCT on the normalized embeddings. When the vectors are normalized the inner product is equal to cosine similarity.
   - For every image, we will perform a K-Nearest Neighbors search (where e.g. $K=5$) to retrieve only the most similar candidates according to the base model.
   - To avoid unnecessary computations we will delete duplicate pairs.
   - We will choose pairs where the similarity is greater than the threshold (e.g. 0.95 like in the paper).
4. From these candidates, we select "blind spots" where at least one reference model (DINOv2 or SigLIP) significantly disagrees (that means the similarity according to this model is lower than e.g. 0.6 (QUESTION: do both reference models need to agree to be considered a blind pair?).

We will execute this procedure three times, rotating the role of the base model.





