# GenAI and Agentic AI: Senior Interview Guide (5 years experience)

At 5 years, interviewers stop asking "what is RAG?" and start asking **"why did you choose that, what broke in production, and how did you measure it?"** Every answer should show three things:

1. **Depth:** you know how it works underneath (tokens, attention, embeddings, retrieval math).
2. **Trade-offs:** you can compare options on quality, latency, cost and risk.
3. **Production scars:** evals, monitoring, failures, security, cost control.

Companion files:
- [genai-system-design.md](genai-system-design.md) has 5 system design questions with model answers.
- [genai_coding_exercises.py](genai_coding_exercises.py) has runnable coding rounds (RAG from scratch, an agent loop, tool calling, evals) with no API key needed.

**Suggested schedule:** Parts 1–3 in days 1–3, Parts 4–5 in days 4–5, Parts 6–7 in day 6, then system design and coding on the following days. Say every Q&A answer **out loud**.

---

## Part 1: LLM fundamentals (the depth check)

### 1.1 Transformer in one minute
- Text → **tokens** (BPE / SentencePiece subwords) → **embeddings** + **positional info** (modern models use **RoPE**).
- A stack of **decoder blocks**, each with **self-attention** + **feed-forward (MLP)**, residual connections and layer norm.
- **Self-attention:** each token creates Query, Key, Value vectors. `Attention(Q,K,V) = softmax(QKᵀ / √d) · V`. The `√d` keeps softmax from saturating.
- **Multi-head attention:** several attention "views" in parallel. **GQA/MQA** (grouped/multi-query attention) share K/V across heads to shrink the KV cache and speed up inference.
- **Causal mask:** a token can only attend to earlier tokens, which makes it autoregressive.
- Output: logits over the vocabulary → **sampling** → next token → repeat.

**Why attention is O(n²):** every token compares with every other token. That's why long context is expensive, and why FlashAttention (IO-aware exact attention), sliding windows and sparse attention exist.

### 1.2 Inference concepts you must explain
| Concept | What to say |
|---|---|
| **Prefill vs decode** | Prefill processes the whole prompt in parallel (compute-bound, drives time-to-first-token). Decode generates one token at a time (memory-bandwidth-bound, drives tokens/sec). |
| **KV cache** | Stores past keys/values so each new token doesn't recompute the whole prefix. Its memory grows with context length × batch, which is the main serving bottleneck. |
| **Prompt caching** | Provider reuses the KV cache for a repeated prompt **prefix**. Put static content (system prompt, tools, docs) first and dynamic content last. Large cost and latency savings. |
| **Continuous batching / PagedAttention** | vLLM-style serving: requests join and leave batches per token, and the KV cache is paged like OS memory. Big throughput gains. |
| **Speculative decoding** | A small draft model proposes tokens, the big model verifies several in one pass. Same output distribution, lower latency. |
| **Quantization** | FP16 → INT8/INT4 (GPTQ, AWQ, GGUF). Less memory, faster, small quality loss. |
| **Temperature / top-p / top-k** | Temperature scales logits (low = deterministic). Top-p samples from the smallest set with cumulative probability p. Use low temperature for extraction/tools, higher for creative work. |
| **Context window** | Max tokens in + out. "Lost in the middle": models use the start and end of long context better than the middle. |

### 1.3 How models are trained (know the pipeline)
1. **Pretraining:** next-token prediction on huge text. Learns language and knowledge.
2. **Supervised fine-tuning (SFT):** instruction/response pairs. Learns to follow instructions.
3. **Preference tuning:** **RLHF** (reward model + PPO), **DPO** (direct optimization on preferred/rejected pairs, simpler, no reward model), **RLAIF / Constitutional AI**.
4. **Reasoning models:** trained with RL on verifiable tasks (math, code) to produce long chains of thought before answering. Trade more tokens/latency for accuracy.

### 1.4 Fine-tuning vs RAG vs prompting (asked in almost every senior loop)
| Need | Best tool |
|---|---|
| Fresh, private, or frequently changing **knowledge** | **RAG** |
| Citations / traceability | **RAG** |
| Consistent **style, format, tone**, or a narrow task | **Fine-tuning** (often LoRA) |
| Lower latency/cost by distilling a big model into a small one | **Fine-tuning / distillation** |
| Quick iteration, most problems | **Prompting first** (+ few-shot, structured output) |

**Senior answer pattern:** "Start with prompting and evals, add RAG for knowledge, fine-tune only when evals show a gap prompting can't close, and you have enough quality data. Fine-tuning teaches *behavior*, not *facts*."

**PEFT:** **LoRA** freezes base weights and trains small low-rank matrices (A·B) added to attention weights, cutting trainable params to under 1%. **QLoRA** = LoRA on a 4-bit quantized base, so you can fine-tune big models on one GPU.

---

## Part 2: Prompting and structured output

- **System prompt:** role, rules, output format, what to do when unsure.
- **Few-shot examples:** show the format and edge cases. Diverse examples beat many similar ones.
- **Chain-of-thought:** "think step by step" or let a reasoning model think. Hide reasoning from end users when needed.
- **Structured output:** use JSON schema / tool-calling / provider "structured outputs" mode, and validate with **Pydantic**. Retry with the validation error fed back on failure.
- **Context engineering** (the 2025–26 term): deciding *what goes into the context window*: instructions, retrieved docs, tool results, memory, history. Most agent quality problems are context problems.
- **Prompt as code:** version prompts, review them, and run evals on every change (regression tests).

**Q: The model sometimes returns invalid JSON. What do you do?**
Use native structured output/tool calling with a schema; validate with Pydantic; on failure, retry once with the error message; set temperature low; keep the schema small and flat; log failures to find patterns.

---

## Part 3: Embeddings, vector search and RAG

### 3.1 Embeddings
- Dense vectors where semantic similarity ≈ **cosine similarity** (dot product on normalized vectors).
- **Bi-encoder** (embed query and doc separately, fast, used for retrieval) vs **cross-encoder** (scores query+doc together, accurate, slow, used for **reranking**).
- Choose by: retrieval benchmark (MTEB) on *your* data, dimensions (storage cost), max input length, multilingual support, hosting.

### 3.2 Vector search
- **Exact (flat)** search is O(n). **ANN** indexes trade a little recall for speed:
  - **HNSW:** layered proximity graph, great recall/speed, memory-heavy. Tune `M`, `ef_construction`, `ef_search`.
  - **IVF:** cluster then search nearest clusters (`nprobe`). Often with **PQ** compression for huge scale.
- Stores: pgvector (if you already run Postgres), Pinecone, Weaviate, Qdrant, Milvus, OpenSearch/Elasticsearch, FAISS (library).
- **Metadata filtering** (tenant, date, ACL) is critical. Pre-filter vs post-filter trade-offs.

### 3.3 The RAG pipeline (and where each stage fails)
**Ingestion:** load → clean → **chunk** → embed → store with metadata.
**Query:** (rewrite) → retrieve → **rerank** → build prompt → generate with citations → (verify).

| Stage | Failure | Fix |
|---|---|---|
| Parsing | tables/PDFs garbled | layout-aware parsers, OCR, convert tables to markdown |
| Chunking | answer split across chunks | recursive/semantic chunking, 10–20% overlap, parent-document retrieval |
| Retrieval | keyword terms (IDs, codes) missed | **hybrid search**: BM25 + vectors, merged with **Reciprocal Rank Fusion** |
| Retrieval | vague user query | **query rewriting**, multi-query, **HyDE** (embed a hypothetical answer) |
| Ranking | right doc at rank 15 | **cross-encoder reranker** over top 50 → keep top 5 |
| Generation | hallucination | "answer only from context, else say you don't know", cite sources, groundedness check |
| Freshness | stale docs | incremental re-indexing, doc versioning, TTL |
| Security | user sees docs they shouldn't | **ACL filtering at retrieval time**, never only in the prompt |

**Chunk size trade-off:** small chunks give precise retrieval but lose context; large chunks keep context but add noise and cost. Typical: 300–800 tokens, then tune with evals.

### 3.4 Advanced RAG patterns
- **Contextual retrieval:** prepend a short LLM-written summary of the doc to each chunk before embedding. Strong recall gains.
- **Parent/child (small-to-big):** retrieve small chunks, return the larger parent section.
- **GraphRAG / knowledge graphs:** for multi-hop questions about relationships across documents.
- **Agentic RAG:** the agent decides *whether* to retrieve, *which* source, and re-queries if results are weak.
- **Long context vs RAG:** long context is simpler for small corpora, but RAG wins on cost, latency, freshness, access control and citations at scale. Often combined.

### 3.5 RAG evaluation
- **Retrieval metrics:** Recall@k, Precision@k, **MRR**, **nDCG**. Requires a labeled set of query → relevant chunk(s).
- **Generation metrics:** **faithfulness/groundedness** (claims supported by context), **answer relevance**, correctness vs reference.
- Tools: RAGAS, DeepEval, TruLens, LangSmith, Arize Phoenix, plus your own LLM-as-judge.
- **Always evaluate retrieval separately from generation**: if retrieval recall is 60%, no prompt will fix answers.

---

## Part 4: Agentic AI

### 4.1 What makes something an agent
An **LLM in a loop** that decides actions, calls **tools**, observes results, and continues until the goal is done. Spectrum (Anthropic's "Building effective agents" framing, worth citing):
- **Workflows** (predefined code paths): prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- **Agents** (model directs its own process): open-ended tasks where steps can't be predicted.

**Senior point:** *use the simplest thing that works.* A deterministic workflow is cheaper, faster and easier to test than an autonomous agent. Add autonomy only where it earns its keep.

### 4.2 Core loop (ReAct)
```
while not done and steps < max_steps:
    response = llm(messages, tools)
    if response has tool_calls:
        for call in tool_calls:
            result = execute(call)          # validate args, timeout, catch errors
            messages.append(tool_result)
    else:
        done = True                         # final answer
```
You'll implement this in [genai_coding_exercises.py](genai_coding_exercises.py).

### 4.3 Tool design (where most agent quality comes from)
- Clear names and descriptions, written like docs for a new engineer. Include when *not* to use the tool.
- Typed, minimal parameters (JSON Schema). Validate inputs before executing.
- Return **concise, useful** results (not 10k-row dumps); paginate; return actionable error messages the model can recover from.
- Make tools **idempotent** where possible; require **confirmation** for destructive actions.
- Fewer, well-designed tools beat many overlapping ones. Large tool sets → tool retrieval/routing.

### 4.4 Planning and reasoning patterns
- **ReAct:** interleave reasoning and acting.
- **Plan-and-execute:** make a plan first, then execute steps, replan on failure. Better for long tasks.
- **Reflection / evaluator-optimizer:** critique and revise the output.
- **Tree of Thoughts / search:** explore multiple branches (expensive, rare in production).

### 4.5 Memory
| Type | Implementation |
|---|---|
| Short-term (working) | conversation messages in context; trim/summarize when long |
| Long-term (semantic) | facts/preferences stored in a vector DB or key-value store, retrieved per turn |
| Episodic | past task trajectories, reused as examples |
| State / checkpoints | graph state persisted per step (e.g. LangGraph checkpointer) for resume and human-in-the-loop |

**Context management:** summarization/compaction of old turns, keep tool results short, offload big artifacts to files and store references.

### 4.6 Multi-agent systems
- **Patterns:** supervisor/orchestrator + workers, hierarchical teams, handoffs (swarm), parallel fan-out then merge, debate/critic.
- **When it helps:** parallelizable research, separation of context (each sub-agent gets a clean context window), different tools/permissions per role.
- **Costs:** many more tokens, harder debugging, coordination failures, compounding errors. **Say this out loud**, because interviewers want to hear you know the downsides.

### 4.7 Frameworks and protocols (know what each is for)
| Name | Use |
|---|---|
| **LangChain** | components: models, prompts, retrievers, tools |
| **LangGraph** | stateful agent **graphs**: nodes, edges, conditional routing, checkpoints, human-in-the-loop, retries. Most common for production agents. |
| **LlamaIndex** | data/RAG-centric framework (ingestion, indexes, query engines) |
| **CrewAI / AutoGen** | role-based multi-agent setups |
| **Provider agent SDKs** (OpenAI Agents SDK, Claude Agent SDK, Google ADK) | built-in agent loops, tools, handoffs, tracing |
| **MCP (Model Context Protocol)** | open standard for connecting models to tools/data via **servers** (tools, resources, prompts). Write a tool server once, use it from any MCP client. |
| **A2A (Agent2Agent)** | protocol for agents from different systems to talk to each other |
| **Pydantic AI / Instructor** | type-safe structured outputs and agents |

**Senior opinion to have ready:** "Frameworks speed up prototyping, but I keep the core loop understandable and avoid heavy abstractions where they hide prompts or make debugging hard. I'd use LangGraph when I need durable state, branching and human approval."

### 4.8 Human-in-the-loop
Approval gates before irreversible actions (payments, emails, deletes), interrupt-and-resume from checkpoints, confidence thresholds for escalation, and editable intermediate outputs.

---

## Part 5: Evaluation (what separates seniors)

- **Build an eval set first:** 50–200 real cases from logs + edge cases + adversarial cases. Grow it from every production failure.
- **Grader types:** exact/regex match, code execution/unit tests, **LLM-as-judge** with a rubric (calibrate against human labels, watch for position/verbosity/self-preference bias), human review.
- **Agent evals:** task success rate, **trajectory** quality (right tools, right order), number of steps, cost and latency per task, tool error rate.
- **Offline vs online:** offline evals gate deploys (CI); online monitors quality on live traffic (thumbs, user edits, escalation rate, A/B tests).
- **Non-determinism:** run each case several times; report pass rate (pass@k / pass^k).

**Q: How do you know your change made the system better?**
"Fixed eval set covering key intents and known failures, run before/after, compare per-category, check for regressions, not just the average; then canary/A-B in production watching task success, cost and latency."

---

## Part 6: Production engineering

### 6.1 Latency
Stream tokens; use smaller/faster models for easy steps (**model routing/cascade**); parallelize independent tool calls and retrievals; prompt caching; shorten prompts and outputs; semantic caching of frequent answers; precompute embeddings; set timeouts.

### 6.2 Cost
Token budgets per request; route by difficulty; cache (prompt + semantic + response); batch APIs for offline jobs; trim context; cap agent steps; track **cost per successful task**, not per call.

### 6.3 Reliability
Retries with exponential backoff + jitter on 429/5xx; fallback models/providers; circuit breakers; timeouts per tool; idempotency keys; queue long jobs; graceful degradation ("I couldn't access X, here's what I found").

### 6.4 Observability
Trace every step: prompt, model, tokens, latency, tool calls, retrieved docs, cost (LangSmith, Langfuse, Arize, OpenTelemetry GenAI conventions). Without traces you can't debug agents.

### 6.5 Security and safety (always asked)
- **Prompt injection:** direct (user) and **indirect** (malicious text inside retrieved docs, web pages, emails, tool outputs). No complete fix, so **defense in depth**:
  - treat all retrieved/tool content as untrusted data, never instructions
  - **least-privilege tools**, scoped credentials, allow-lists
  - human approval for sensitive actions
  - separate the "reads untrusted content" agent from the "can take powerful actions" agent
  - input/output classifiers, output validation
- **Data leakage:** PII redaction, tenant isolation, ACLs at retrieval, don't log secrets.
- **Guardrails:** input moderation, topic restriction, output checks (PII, toxicity, groundedness), schema validation (e.g. Guardrails AI, NeMo Guardrails, provider moderation).
- Know the **OWASP Top 10 for LLM Applications** headings: prompt injection, sensitive info disclosure, supply chain, data/model poisoning, improper output handling, excessive agency, system prompt leakage, vector/embedding weaknesses, misinformation, unbounded consumption.

### 6.6 Hallucination: the full answer
Ground with RAG + "say I don't know"; require citations and verify cited spans exist; low temperature; structured output; self-consistency or verifier model; groundedness evals; show sources to users; measure the rate.

---

## Part 7: Q&A bank (answer out loud, 1–2 minutes each)

**Fundamentals**
1. Explain self-attention and why it's divided by √d.
2. What's the KV cache and why does it limit batch size?
3. Prefill vs decode, and how each affects TTFT and throughput.
4. How do temperature and top-p differ?
5. What is LoRA and why does it work? (Weight updates are low-rank.)
6. RLHF vs DPO?
7. Why do LLMs hallucinate? (Trained to produce plausible text, not to verify truth; knowledge gaps; no grounding.)
8. What are reasoning models and when would you *not* use one? (Latency/cost-sensitive simple tasks.)

**RAG**
9. Walk me through a production RAG pipeline end to end.
10. How do you choose chunk size and overlap?
11. Why hybrid search? Explain Reciprocal Rank Fusion. (`score = Σ 1/(k + rank)`, k≈60.)
12. Bi-encoder vs cross-encoder?
13. HNSW vs IVF?
14. Retrieval looks right but answers are wrong. How do you debug? (Check retrieved context in traces → is the answer in it? → prompt/ordering/lost-in-middle → generation eval.)
15. How do you enforce document permissions in RAG?
16. RAG vs long context?
17. How do you evaluate RAG? Name retrieval and generation metrics.

**Agents**
18. Workflow vs agent. When do you choose each?
19. Implement/describe a ReAct loop. How do you stop infinite loops? (Max steps, budget, repeated-action detection, timeouts.)
20. What makes a good tool definition?
21. How do agents handle tool errors?
22. Explain agent memory types and how you'd manage a long conversation.
23. When is multi-agent worth it, and what are the costs?
24. What is MCP and why does it matter?
25. LangGraph concepts: state, nodes, edges, conditional edges, checkpointer, interrupt.
26. How do you evaluate an agent?
27. How would you add human approval before an agent sends an email?

**Production and security**
28. How do you cut LLM cost by 50% without hurting quality?
29. How do you reduce latency for a chat assistant?
30. What is prompt injection, and how do you defend an agent that reads emails?
31. How do you monitor an LLM app in production?
32. How do you handle provider outages and rate limits?
33. How do you version and roll out prompt changes safely?
34. How do you handle PII?

**Senior/leadership**
35. Tell me about a GenAI system you shipped: architecture, metrics, what failed.
36. A stakeholder wants "an AI agent for everything." How do you scope it?
37. How do you decide build vs buy (vendor API vs open-source model)? (Quality, cost at scale, latency, data privacy, ops burden, customization.)
38. How do you mentor engineers new to GenAI?

### Model answers for the most common five

**Q9 Production RAG pipeline.** "Ingestion: parse with layout-aware loaders, clean, chunk around 500 tokens with overlap and keep headings as metadata, embed, store in a vector DB with tenant and ACL metadata, and reindex incrementally on change. Query: rewrite conversational queries into standalone ones, run hybrid BM25 + vector search with ACL filters, merge with RRF, rerank the top 50 with a cross-encoder, keep 5, then generate with instructions to answer only from context and cite chunk IDs. Around it: prompt caching for the system prompt, streaming, tracing every step, and an eval set measuring recall@5 and faithfulness, run in CI on every change."

**Q19 Infinite loops.** "Hard cap on steps and tokens, detect repeated identical tool calls, timeouts per tool and per task, make errors informative so the model can change course, and a fallback that returns partial results or escalates to a human."

**Q28 Cut cost 50%.** "Measure first: cost per successful task, broken down by step. Then: route easy requests to a small model, prompt-cache the static prefix, trim retrieved context with a reranker, cap output length, semantic-cache repeated questions, use batch APIs for offline jobs. Verify every change against the eval set so quality doesn't drop."

**Q30 Prompt injection for an email agent.** "Treat email content as untrusted data, clearly delimited. The reading agent has read-only tools; sending or forwarding requires a separate privileged step with user approval. Allow-list recipients and domains, strip or flag hidden text, classify inputs for injection, validate outputs, and log everything. Accept it can't be fully prevented, so limit blast radius."

**Q35 Your project (use STAR + metrics).** Prepare 2 stories. Template:
- **Situation:** business problem and scale (users, docs, requests/day).
- **Task:** your role and the goal metric.
- **Action:** architecture choices *and alternatives you rejected and why*; how you built evals; one hard production problem you solved.
- **Result:** numbers (accuracy 72→89%, latency p95 6s→2.1s, cost −40%, tickets deflected 30%).
- **Learning:** what you'd do differently.

---

## Final-week checklist
- [ ] Can draw a RAG and an agent architecture on a whiteboard in 5 minutes
- [ ] Can code a ReAct loop and cosine-similarity retrieval from scratch
- [ ] Can explain attention, KV cache, LoRA, and RRF without notes
- [ ] Have 2 project stories with numbers
- [ ] Have opinions: when *not* to use agents, multi-agent, fine-tuning
- [ ] Know prompt injection defenses and the OWASP LLM list
- [ ] Done all 5 system designs in [genai-system-design.md](genai-system-design.md) out loud, 35 minutes each
