# 🔄 Models Execution Order - Visual Guide

**Quick Reference**: Execution order of models in the RAG system  
**Version**: v6.0  
**Date**: November 23, 2025

---

## 📋 Quick Summary

```
SETUP PHASE (once per manual):
    Dolphin OCR → E5-Large → CLIP → Databases ready

QUERY PHASE (every user question):
    E5-Large → ChromaDB → CLIP → Gemini Flash → Gemini Judge
```

---

## 🏗️ PHASE 1: SETUP (BLOCK 0)

### Execution: Once per PDF manual

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: PANDA.pdf manual (250 pages)                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  MODEL #1: DOLPHIN OCR                ┃
        ┃  (GOT-OCR2_0)                         ┃
        ┃                                       ┃
        ┃  What it does:                        ┃
        ┃  • Reads each PDF page                ┃
        ┃  • Extracts text (OCR)                ┃
        ┃  • Identifies and crops images        ┃
        ┃  • Maintains document structure       ┃
        ┃                                       ┃
        ┃  Parameters:                          ┃
        ┃  • batch_size: 1 page at a time       ┃
        ┃  • device: CUDA (GPU)                 ┃
        ┃  • dpi: 300 (high resolution)         ┃
        ┃                                       ┃
        ┃  Time: ~5 minutes (250 pages)         ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          ↓
        ┌─────────────────────────────────────┐
        │  OUTPUT:                            │
        │  • 2,541 text chunks                │
        │  • 850 PNG images                   │
        │  • Metadata (page, bbox, etc.)      │
        └─────────────────────────────────────┘
                          ↓
                ┌─────────┴─────────┐
                ↓                   ↓
┏━━━━━━━━━━━━━━━━━━━━┓   ┏━━━━━━━━━━━━━━━━━━━━┓
┃ MODEL #2:          ┃   ┃ MODEL #3:          ┃
┃ E5-LARGE           ┃   ┃ CLIP               ┃
┃ (Text Embedder)    ┃   ┃ (Image Embedder)   ┃
┃                    ┃   ┃                    ┃
┃ What it does:      ┃   ┃ What it does:      ┃
┃ • Converts text    ┃   ┃ • Converts images  ┃
┃   to vectors       ┃   ┃   to vectors       ┃
┃ • 1024 dimensions  ┃   ┃ • 512 dimensions   ┃
┃ • L2 normalization ┃   ┃ • L2 normalization ┃
┃                    ┃   ┃                    ┃
┃ Input:             ┃   ┃ Input:             ┃
┃ • 2,541 chunks     ┃   ┃ • 850 images       ┃
┃                    ┃   ┃ • Quality filtering┃
┃ Parameters:        ┃   ┃                    ┃
┃ • batch_size: 32   ┃   ┃ Parameters:        ┃
┃ • device: cuda     ┃   ┃ • size: 224×224    ┃
┃ • fp16: True       ┃   ┃ • device: cuda     ┃
┃                    ┃   ┃ • fp16: True       ┃
┃ Time: ~2 min       ┃   ┃                    ┃
┃   (all chunks)     ┃   ┃ Time: ~3 min       ┃
┃                    ┃   ┃   (595 quality)    ┃
┗━━━━━━━━━━━━━━━━━━━━┛   ┗━━━━━━━━━━━━━━━━━━━━┛
        ↓                          ↓
        ↓                          ↓
┌───────────────────┐   ┌───────────────────┐
│  ChromaDB (Text)  │   │ ChromaDB (Images) │
│  2,541 vectors    │   │ 595 vectors       │
│  [1024-dim]       │   │ [512-dim]         │
└───────────────────┘   └───────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ✅ SETUP COMPLETE                                           │
│  Databases ready for queries                                │
│  Total time: ~10 minutes                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 PHASE 2: QUERY (BLOCK 3)

### Execution: Every user question

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: "How does ASR work on PANDA?"                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  STEP 1: PREPROCESSING                ┃
        ┃  (Not an AI model)                    ┃
        ┃                                       ┃
        ┃  • PromptProtection: sanitize query   ┃
        ┃  • VehicleDetector: find "PANDA"     ┃
        ┃                                       ┃
        ┃  Time: <1ms                           ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  MODEL #2: E5-LARGE (REUSED)          ┃
        ┃  Text Embedding                       ┃
        ┃                                       ┃
        ┃  What it does:                        ┃
        ┃  • Converts query to vector           ┃
        ┃  • Same model used for indexing       ┃
        ┃                                       ┃
        ┃  Input: "How does ASR work on PANDA?" ┃
        ┃  Output: vector [1024 dimensions]    ┃
        ┃                                       ┃
        ┃  Time: 22ms                           ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          ↓
        ┌─────────────────────────────────────┐
        │  CHROMADB TEXT SEARCH               │
        │  (Not an AI model)                  │
        │                                     │
        │  • Calculate cosine similarity      │
        │  • Query vs all 2,541 chunks        │
        │  • Filter by manual="PANDA"         │
        │  • Sort by similarity               │
        │  • Return top 30                    │
        │                                     │
        │  Time: 6ms                          │
        └─────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────┐
        │  STAGE 1 COMPLETE                   │
        │  Top 30 text chunk candidates       │
        └─────────────────────────────────────┘
                          ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  MODEL #3: CLIP (REUSED)              ┃
        ┃  Visual Re-ranking                    ┃
        ┃                                       ┃
        ┃  What it does:                        ┃
        ┃  • Encode query as text (CLIP)        ┃
        ┃  • Find images in pages of            ┃
        ┃    the 30 chunks                      ┃
        ┃  • Calculate visual similarity        ┃
        ┃  • Combine with text similarity       ┃
        ┃                                       ┃
        ┃  Hybrid Score Formula:                ┃
        ┃  score = 0.55×CLIP + 0.45×text       ┃
        ┃                                       ┃
        ┃  Input:                               ┃
        ┃  • Query text: "How does ASR..."      ┃
        ┃  • ~18 candidate images               ┃
        ┃                                       ┃
        ┃  Output: Top 6 images + chunks        ┃
        ┃                                       ┃
        ┃  Time: 105ms                          ┃
        ┃  • Text encode: 8ms                   ┃
        ┃  • Image similarity: 97ms             ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          ↓
        ┌─────────────────────────────────────┐
        │  QUALITY FILTER                     │
        │  (Not an AI model)                  │
        │                                     │
        │  Checks:                            │
        │  • File size ≥ 10KB                 │
        │  • Resolution ≥ 150×150px           │
        │  • Aspect ratio ≤ 6.0               │
        │                                     │
        │  Input: 6 images                    │
        │  Output: 4 images pass ✅           │
        │                                     │
        │  Time: 3ms                          │
        └─────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────┐
        │  STAGE 2 COMPLETE                   │
        │  • 6 final text chunks              │
        │  • 4 quality images                 │
        └─────────────────────────────────────┘
                          ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  MODEL #4: GEMINI 2.0 FLASH LITE      ┃
        ┃  RAG Generation                       ┃
        ┃                                       ┃
        ┃  What it does:                        ┃
        ┃  • Receives 6 chunks as context       ┃
        ┃  • Receives user query                ┃
        ┃  • Generates natural language         ┃
        ┃    response                           ┃
        ┃  • Grounded response (no hallucination)┃
        ┃                                       ┃
        ┃  Input:                               ┃
        ┃  • System prompt: "Answer based on..."┃
        ┃  • Context: [6 chunks, ~1,500 tokens] ┃
        ┃  • Question: "How does ASR work..."   ┃
        ┃                                       ┃
        ┃  Parameters:                          ┃
        ┃  • temperature: 0.1 (deterministic)   ┃
        ┃  • max_tokens: 512                    ┃
        ┃                                       ┃
        ┃  Output:                              ┃
        ┃  "The ASR system prevents wheel       ┃
        ┃   spin during acceleration..."        ┃
        ┃  (~85 tokens)                         ┃
        ┃                                       ┃
        ┃  Time: 1,050ms                        ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          ↓
        ┌─────────────────────────────────────┐
        │  CONFIDENCE CALCULATION             │
        │  (Not an AI model)                  │
        │                                     │
        │  Multi-source confidence:           │
        │  • Retrieval quality: 0.91          │
        │  • Context relevance: 0.84          │
        │  • Answer quality: 0.70             │
        │                                     │
        │  Aggregate: 0.83 (HIGH) ✅          │
        │                                     │
        │  Time: 5ms                          │
        └─────────────────────────────────────┘
                          ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  MODEL #5: GEMINI 2.0                 ┃
        ┃  LLM Judge (OPTIONAL)                 ┃
        ┃                                       ┃
        ┃  What it does:                        ┃
        ┃  • Evaluates response quality         ┃
        ┃  • 3 dimensions:                      ┃
        ┃    1. Faithfulness (context grounding)┃
        ┃    2. Relevance (query addressing)    ┃
        ┃    3. Completeness (comprehensive)    ┃
        ┃                                       ┃
        ┃  Input:                               ┃
        ┃  • Original query                     ┃
        ┃  • Context (6 chunks)                 ┃
        ┃  • Generated response                 ┃
        ┃                                       ┃
        ┃  Output (JSON):                       ┃
        ┃  {                                    ┃
        ┃    "faithfulness": 5,                 ┃
        ┃    "relevance": 4,                    ┃
        ┃    "completeness": 4,                 ┃
        ┃    "average": 4.33                    ┃
        ┃  }                                    ┃
        ┃                                       ┃
        ┃  Parameters:                          ┃
        ┃  • temperature: 0.0 (deterministic)   ┃
        ┃  • max_tokens: 500                    ┃
        ┃                                       ┃
        ┃  Time: 2,600ms                        ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          ↓
        ┌─────────────────────────────────────┐
        │  FINAL RESPONSE PACKAGE             │
        │                                     │
        │  • Answer text                      │
        │  • 4 quality images                 │
        │  • Confidence: 0.83 (HIGH)          │
        │  • Judge scores: 4.33/5             │
        │  • Complete metrics                 │
        └─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ✅ QUERY COMPLETE                                           │
│  Total time: 3.8s (with judge), 1.2s (without judge)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Visual Timeline

```
SETUP PHASE (once):
│
├─► [0-5min]   Dolphin OCR: PDF → Text + Images
├─► [5-7min]   E5-Large: Text → Embeddings (1024-dim)
├─► [7-10min]  CLIP: Images → Embeddings (512-dim)
└─► [10min]    ✅ Databases ready

QUERY PHASE (every request):
│
├─► [0ms]      Preprocessing (VehicleDetector)
├─► [22ms]     E5-Large: Query → Embedding
├─► [28ms]     ChromaDB: Text search (top 30)
├─► [133ms]    CLIP: Visual re-ranking (top 6)
├─► [136ms]    Quality filter (4 images pass)
├─► [1,186ms]  Gemini Flash: Generate answer
├─► [1,191ms]  ConfidenceCalculator
├─► [3,791ms]  Gemini Judge: Evaluate (optional)
└─► [3,791ms]  ✅ Response ready

Average time without judge: 1.2s
Average time with judge: 3.8s
```

---

## 🔢 Numerical Order

### Setup Phase
1. **Dolphin OCR** (GOT-OCR2_0)
2. **E5-Large** (multilingual-e5-large)
3. **CLIP** (openai/clip-vit-base-patch32)

### Query Phase (Runtime)
1. **Preprocessing** (VehicleDetector, PromptProtection)
2. **E5-Large** (REUSED: query embedding)
3. **ChromaDB** (similarity search)
4. **CLIP** (REUSED: visual re-ranking)
5. **ImageQualityFilter** (quality check)
6. **Gemini Flash Lite** (answer generation)
7. **ConfidenceCalculator** (aggregate metrics)
8. **Gemini 2.0** (OPTIONAL: LLM Judge)

---

## 🎯 Key Points for Presentation

### Quick Explanation (1 minute)

**Setup Phase**:
> "Before we can answer questions, we prepare the database using 3 models in sequence: **Dolphin OCR** reads the 250-page PDF and extracts 2,541 text blocks and 850 images. Then **E5-Large** converts all text into 1024-dimensional vectors representing semantic meaning. Finally, **CLIP** converts images into 512-dimensional vectors. Everything is stored in ChromaDB vector databases. This takes about 10 minutes but only happens once per manual."

**Query Phase**:
> "When a user asks a question, 4 main steps occur:
> 
> 1. **E5-Large** (reused) converts the question into a vector and finds the 30 most semantically similar text chunks in the database (28ms)
> 
> 2. **CLIP** (reused) analyzes images in candidate pages and re-ranks them by combining text similarity (45%) and visual similarity (55%), selecting the top 6 results (105ms)
> 
> 3. **Gemini Flash Lite** generates a natural language answer based only on the 6 retrieved chunks, avoiding hallucinations (1050ms)
> 
> 4. Optionally, **Gemini 2.0** evaluates the quality on 3 criteria: faithfulness to context, relevance to question, and completeness (2600ms)
> 
> Total time: 1.2 seconds without evaluation, or 3.8 seconds with complete evaluation."

### Detailed Explanation (3 minutes)

**Setup Phase (10 minutes total)**:
> "The setup phase uses 3 AI models to transform a raw PDF into a searchable database. First, **Dolphin OCR** with the GOT-OCR2_0 architecture processes each page using a Vision Transformer encoder. It identifies text regions, performs OCR with 98%+ accuracy on automotive technical content, and extracts images with their bounding boxes and surrounding context. This produces 2,541 text chunks and 850 PNG images.
>
> Next, **E5-Large** (multilingual-e5-large), a 560M parameter sentence transformer, embeds all text chunks. It uses a 24-layer XLM-RoBERTa encoder to create 1024-dimensional dense vectors that capture semantic meaning. These vectors are L2-normalized and stored in ChromaDB for efficient similarity search.
>
> Finally, **CLIP ViT-B/32**, a 151M parameter dual-encoder model, processes the images. After quality filtering (resolution ≥150×150px, size ≥10KB, aspect ratio ≤6.0), 595 high-quality images are embedded into 512-dimensional vectors in the same semantic space as text. This allows us to search both text and images using natural language queries."

**Query Phase (1.2-3.8 seconds)**:
> "When a user asks a question, the system executes a sophisticated two-stage retrieval process:
>
> **Stage 1 - Text Retrieval (28ms)**: E5-Large embeds the query using the same model that indexed the documents, ensuring consistency. ChromaDB then performs a cosine similarity search across all 2,541 chunks, filtering by detected vehicle if applicable. The top 30 candidates are selected based on semantic similarity.
>
> **Stage 2 - Visual Re-ranking (105ms)**: CLIP encodes the query text through its text encoder, producing a 512-dimensional vector. For each image in the candidate pages, CLIP computes visual similarity. A hybrid score combines text similarity (45%) and visual similarity (55%) - this weight was empirically optimized on a 50-query validation set. The top 6 results balance both semantic and visual relevance.
>
> After quality filtering removes low-resolution or oddly-shaped images, **Gemini 2.0 Flash Lite** generates the answer. Using a carefully crafted RAG prompt with low temperature (0.1) for deterministic output, it produces a grounded response in about 1 second. The model attends to the retrieved context and avoids hallucinations by explicitly instructing it to use only provided information.
>
> Finally, an optional **LLM-as-a-Judge** evaluation uses Gemini 2.0 to score the response on faithfulness (grounding in context), relevance (addressing the query), and completeness (comprehensive answer). This methodology, based on recent NeurIPS 2023 research, achieves 86% agreement with human expert evaluations."

### Why This Architecture?

✅ **Multi-modal**: CLIP unites text and images in the same space  
✅ **Two-stage**: Fast text search (E5) refined by precise visual ranking (CLIP)  
✅ **Grounded**: Gemini Flash uses ONLY retrieved context  
✅ **Evaluated**: Gemini Judge provides objective metrics (86% human agreement)  
✅ **SOTA 2025**: State-of-the-art techniques validated by research

### Results

- **Image F1**: 0.89 (CLIP re-ranking excellence)
- **Text F1**: 0.75 (E5-Large retrieval)
- **Confidence**: 0.81 average (68% HIGH)
- **LLM Judge**: 4.37/5 average
- **User Satisfaction**: 4.6/5 (+64% from v0_whatsapp)
- **Response Time**: 1.2s (production-ready)

---

## 📝 Simplified Nomenclature

To make the presentation clearer, you can use these simplified names:

| Technical Name | Simple Name | Function |
|----------------|-------------|----------|
| Dolphin OCR (GOT-OCR2_0) | "PDF Reader" | Extracts text and images |
| multilingual-e5-large | "Text Embedder" | Converts text to numbers |
| CLIP ViT-B/32 | "Visual Embedder" | Converts images to numbers |
| Gemini 2.0 Flash Lite | "Answer Generator" | Creates the final response |
| Gemini 2.0 | "Quality Judge" | Evaluates responses |

---

## 🔄 Complete Flow Diagram

```
SETUP (Once)                    QUERY (Every time)
─────────────────               ───────────────────

PDF Manual                      User Question
    ↓                               ↓
[Dolphin OCR]                   Preprocessing
    ↓                               ↓
Text + Images                   [E5-Large]
    ↓                               ↓
┌───┴────┐                      Query Vector
↓        ↓                           ↓
[E5]   [CLIP]                   ChromaDB Search
↓        ↓                           ↓
Text   Image                    Top 30 Chunks
DB     DB                            ↓
                                [CLIP Re-rank]
                                    ↓
                                Top 6 + Images
                                    ↓
                                Quality Filter
                                    ↓
                                4-6 Images
                                    ↓
                                [Gemini Flash]
                                    ↓
                                Answer
                                    ↓
                                Confidence
                                    ↓
                                [Gemini Judge]
                                    ↓
                                Complete Response
```

---

## 💡 Teaching Tips

### For Technical Audience

Focus on:
- **Architecture decisions**: Why two-stage? Why CLIP weight 0.55?
- **Empirical validation**: 50-query validation set, 100-image threshold analysis
- **Performance optimization**: FP16, batch processing, GPU memory management
- **Evaluation methodology**: LLM-as-Judge, multi-source confidence

### For Non-Technical Audience

Use analogies:
- **E5-Large**: "Like a librarian who remembers where every topic is discussed"
- **CLIP**: "Like someone who can look at a picture and understand what it shows"
- **Gemini Flash**: "Like an expert who reads the manual and explains it to you"
- **Gemini Judge**: "Like a teacher grading homework on accuracy and completeness"

### Common Questions

**Q: Why not use just one big model?**
A: Specialized models excel at specific tasks. CLIP is best at visual-semantic matching, E5-Large is best at text similarity, Gemini is best at natural language generation. Combining them gives better results than any single model.

**Q: Why 1024 dimensions for text but only 512 for images?**
A: Text has more semantic nuance requiring higher dimensionality. Images can be effectively represented in 512 dimensions for matching purposes. These dimensions were chosen by the model creators based on extensive research.

**Q: How do you avoid hallucinations?**
A: By using RAG (Retrieval-Augmented Generation) with strict prompting. We explicitly instruct Gemini to use ONLY the retrieved context and to say "I don't know" if information isn't present. The low temperature (0.1) also makes output more deterministic and faithful.

**Q: Why use Gemini instead of open-source models?**
A: Gemini Flash Lite offers the best balance of speed (<1s), quality (near GPT-4), and cost (10x cheaper than full Gemini). For production deployment serving many users, this combination is optimal. Open-source alternatives would require expensive GPU infrastructure.

---

**Document Version**: 2.0  
**Last Updated**: November 23, 2025  
**Language**: English  
**Companion Document**: MODELS_DEEP_DIVE.md
