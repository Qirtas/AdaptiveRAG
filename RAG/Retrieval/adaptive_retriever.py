"""
adaptive_retriever.py - Adaptive Retrieval Pipeline

"""

import numpy as np
from typing import List, Tuple, Optional, Union, Set
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from sentence_transformers import CrossEncoder
import time
import re
from sentence_transformers import SentenceTransformer
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveRetriever:
    """
    Adaptive retriever for dense-only retrieval with ChromaDB.
    """

    def __init__(
            self,
            vectorstore: Chroma,
            k_init: int = 24,
            pool_cap: int = 20
    ):
        """
        Initialize the adaptive retriever.

        Args:
            vectorstore: ChromaDB vector store with document embeddings
            k_init: Number of documents to retrieve (default 24)
            pool_cap: Maximum candidates to keep for Step 2 (default 20)
        """
        self.vectorstore = vectorstore
        self.k_init = k_init
        self.pool_cap = pool_cap

        logger.info("AdaptiveRetriever initialized (Dense-only mode)")
        print(f"\nAdaptiveRetriever Configuration:")
        print(f"  - k_init: {k_init} (documents to retrieve)")
        print(f"  - pool_cap: {pool_cap} (max candidates for Step 2)")
        print(f"  - Mode: Dense retrieval only\n")

    def _extract_entities_simple(self, query: str) -> List[str]:
        """
        Simple entity extraction based on capitalization and known patterns.
        Minimal implementation that catches most domain entities.
        """
        import re
        entities = []

        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, query)

        for match in matches:
            # Filter out common non-entities
            if match.lower() not in {'explain', 'how', 'what', 'the', 'in'}:
                entities.append(match)

        # Also check for "between X and Y" pattern
        between_pattern = r'between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|when|while|for)|[?.!,;]|$)'
        for match in re.finditer(between_pattern, query, re.IGNORECASE):
            e1 = match.group(1).strip()
            e2 = match.group(2).strip()
            # Clean up common trailing words
            e1 = re.sub(r'\s+(in|when|while|for)$', '', e1, flags=re.IGNORECASE).strip()
            e2 = re.sub(r'\s+(in|when|while|for)$', '', e2, flags=re.IGNORECASE).strip()
            if e1:
                entities.append(e1)
            if e2:
                entities.append(e2)

        vs_pattern = r'\b(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s+(?:in|when|while|for)|[?.!,;]|$)'
        for match in re.finditer(vs_pattern, query, re.IGNORECASE):
            e1 = match.group(1).strip()
            e2 = match.group(2).strip()
            e1 = re.sub(r'\s+(in|when|while|for)$', '', e1, flags=re.IGNORECASE).strip()
            e2 = re.sub(r'\s+(in|when|while|for)$', '', e2, flags=re.IGNORECASE).strip()
            if e1 and len(e1.split()) <= 5:
                entities.append(e1)
            if e2 and len(e2.split()) <= 5:
                entities.append(e2)

        # De-duplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique_entities.append(e)

        noise_words = {'if', 'when', 'where', 'then', 'else', 'while', 'for',
                       'compare', 'explain', 'describe', 'what', 'how', 'why',
                       'which', 'tell', 'show', 'define'}
        unique_entities = [e for e in unique_entities if e.lower() not in noise_words]

        return unique_entities

    def step1_wide_retrieval(self, query: str) -> List[Tuple[Document, float]]:
        """
        Step 1 (revised): Wide retrieval with multi-query fan-out and score fusion.

        - Normalizes per-list scores, then fuses with weights + an RRF term.
        - Adds a co-mention bonus for docs that mention *all* entities.
        - Applies lightweight priors (prefer KPI; match subfamilies hinted by query).
        - Coverage: ensure at least one doc per detected entity in top-K.

        Returns:
            List of (Document, fused_score) sorted by fused_score desc, capped to pool_cap.
        """
        import time
        import numpy as np

        # ----------------------------
        # helpers + config (scoped)
        # ----------------------------
        WEIGHTS_RELATION = {
            "full": 0.9,
            "name_boost": 1.8,
            "entity": 1.5,  # base for each entity list
            "pair": 2.0,
            "rel": 2.0,
            "vs": 2.0,
        }
        WEIGHTS_DEFAULT = {
            "full": 1.0,
            "name_boost": 1.4,
            "entity": 1.3,
            "pair": 1.2,
            "rel": 1.2,
            "vs": 1.2,
        }
        TYPE_PRIOR = {"KPI": 1.08, "Objective": 1.0, "Criteria": 1.0}
        SUBFAMILY_PRIOR = {
            # lowercase hint -> multiplier
            "operational efficiency": 1.10,
            "data quality": 1.08,
            "data monetization": 1.06,
        }
        RRF_K = 60
        RRF_FACTOR = 0.5
        CO_MENTION_BONUS = 0.15  # added after weighted sum
        TOP_PRINT = 20

        def _get_key(doc: Document) -> str:
            if hasattr(self, "_get_document_key"):
                try:
                    return self._get_document_key(doc)
                except Exception:
                    pass
            t = doc.metadata.get("type", "")
            n = doc.metadata.get("name", "")
            if t and n:
                return f"{t}::{n}"
            # stable content hash fallback
            import hashlib
            content_hash = hashlib.sha256((doc.page_content or "").encode("utf-8")).hexdigest()[:16]
            return f"content::{content_hash}"

        def _normalize_list(scores_dict: dict) -> dict:
            if not scores_dict:
                return scores_dict
            vals = list(scores_dict.values())
            vmin, vmax = min(vals), max(vals)
            if vmax <= vmin:
                return {k: 0.0 for k in scores_dict}
            return {k: (v - vmin) / (vmax - vmin) for k, v in scores_dict.items()}

        def _contains_any(text_lc: str, terms: set) -> bool:
            return any(t in text_lc for t in terms if t)

        def _alts_for(entity: str, alt_map: dict) -> set:
            s = {entity.lower()}
            s |= set(a.lower() for a in alt_map.get(entity.lower(), []))
            return s

        def _doc_mentions_entity(doc: Document, entity: str, alt_map: dict) -> bool:
            terms = _alts_for(entity, alt_map)
            text = (doc.page_content or "").lower()
            meta = " ".join(str(v) for v in doc.metadata.values()).lower()
            blob = text + " " + meta
            return _contains_any(blob, terms)

        def _build_alt_map_lazy(docs: List[Document]) -> dict:

            alt_map = {}
            for d in docs:
                name = (d.metadata.get("name") or "").strip().lower()
                if not name:
                    continue
                raw = d.metadata.get("alternative_names")
                if not raw:
                    continue
                # Try to parse simple list-like strings: "['A', 'B']" or "[\\A, \\B]" etc.
                txt = str(raw)
                # Remove brackets and slashes, then split by comma
                cleaned = txt.strip().strip("[]").replace("\\", "")
                alts = [a.strip(" '\"") for a in cleaned.split(",") if a.strip(" '\"")]
                if alts:
                    alt_map.setdefault(name, set()).update(alts)
            # cast sets to lists
            return {k: sorted(list(v)) for k, v in alt_map.items()}

        def _search_list(list_name: str, q: str, k: int) -> list:

            hits = []
            try:
                res = self.vectorstore.similarity_search_with_relevance_scores(q, k=k)
                for rank, (doc, score) in enumerate(res):
                    hits.append((doc, float(score), rank))
            except Exception:
                # Fallback to distance -> similarity
                res = self.vectorstore.similarity_search_with_score(q, k=k)
                for rank, (doc, dist) in enumerate(res):
                    score = 1.0 - float(dist)
                    hits.append((doc, score, rank))
            return hits


        print(f"\n{'=' * 60}")
        print(f"STEP 1: WIDE RETRIEVAL - Query: '{query}'")
        print(f"{'=' * 60}")
        t0 = time.time()

        query_lower = query.lower()
        entities = []
        if hasattr(self, "_extract_entities"):
            try:
                entities = sorted(list(self._extract_entities(query)))
            except Exception:
                entities = []
        # keep the first two as primary if many
        primary = entities[:2] if entities else []

        # ----------------------------
        # build sub-queries
        # ----------------------------
        subqs: dict[str, list[str]] = {}

        # full query as-is
        subqs["full"] = [query]

        entities = self._extract_entities_simple(query)
        if entities:
            subqs["entity"] = entities

        name_boost_terms = []
        # reuse entities for name_boost too
        name_boost_terms.extend(primary if primary else entities)

        import re as _re
        quoted = _re.findall(r'"([^"]+)"', query)
        name_boost_terms.extend([q.strip() for q in quoted if q.strip()])
        name_boost_terms = list({t for t in name_boost_terms if t})
        if name_boost_terms:
            subqs["name_boost"] = name_boost_terms

        entity_lists = []
        for e in primary if primary else entities:
            entity_lists.append(e)
            # also test variant with trailing punctuation
            entity_lists.append(f"{e}.")
        if entity_lists:
            subqs["entity"] = entity_lists

        if len(primary) >= 2:
            a, b = primary[0], primary[1]
            subqs["pair"] = [f"{a} {b}", f"{b} {a}"]
            subqs["rel"] = [f"relation between {a} and {b}",
                            f"relationship of {a} to {b}",
                            f"how are {a} and {b} related?"]
            subqs["vs"] = [f"{a} vs {b}", f"{b} vs {a}"]

        # ----------------------------
        # run all searches
        # ----------------------------
        hits_by_list: dict[str, list] = {}
        total_raw_hits = 0

        per_list_k = max(self.k_init, self.pool_cap)  # keep it reasonably wide per sub-query

        # gather hits and accumulate for alt-map building
        all_docs_touched: list[Document] = []

        for list_name, queries in subqs.items():
            hits_by_list[list_name] = []
            for q in queries:
                hits = _search_list(list_name, q, per_list_k)
                hits_by_list[list_name].extend(hits)
                total_raw_hits += len(hits)
                all_docs_touched.extend([d for (d, _, _) in hits])

        # unique candidates encountered
        uniq_keys = set()
        for lst in hits_by_list.values():
            for (d, _, _) in lst:
                uniq_keys.add(_get_key(d))


        alt_map = {}
        try:
            alt_map = _build_alt_map_lazy(all_docs_touched)
        except Exception:
            alt_map = {}

        # ----------------------------
        # choose weights profile
        # ----------------------------
        is_relation = any(x in query_lower for x in ["relation", "between", "trade-off", "tradeoff", "versus", " vs "])
        W = WEIGHTS_RELATION if is_relation else WEIGHTS_DEFAULT

        # ----------------------------
        # per-list normalization + weighted fusion
        # ----------------------------
        # reduce duplicates per list with best score; then normalize list
        norm_lists: dict[str, dict] = {}
        for list_name, hits in hits_by_list.items():
            tmp = {}
            for doc, raw, rank in hits:
                key = _get_key(doc)
                tmp[key] = max(tmp.get(key, 0.0), float(raw))
            norm_lists[list_name] = _normalize_list(tmp)

        fused: dict[str, float] = {}
        provenance: dict[str, set] = {}

        # weighted sum
        for list_name, scores in norm_lists.items():
            weight = W.get(list_name, 1.0)
            for key, s in scores.items():
                fused[key] = fused.get(key, 0.0) + weight * s
                provenance.setdefault(key, set()).add(list_name)

        # RRF safety term
        for list_name, hits in hits_by_list.items():
            for doc, _, rank in hits:
                key = _get_key(doc)
                rrf = 1.0 / (RRF_K + rank + 1)
                fused[key] = fused.get(key, 0.0) + RRF_FACTOR * rrf
                provenance.setdefault(key, set()).add(f"rrf:{list_name}")


        key2doc: dict[str, Document] = {}
        for hits in hits_by_list.values():
            for doc, _, _ in hits:
                key2doc.setdefault(_get_key(doc), doc)

        for key, score in list(fused.items()):
            doc = key2doc.get(key)
            if not doc:
                continue
            t = doc.metadata.get("type", "")
            fused[key] = fused[key] * TYPE_PRIOR.get(t, 1.0)

            subf = (doc.metadata.get("bsc_subfamilies") or "").lower()
            for hint, mult in SUBFAMILY_PRIOR.items():
                if hint in query_lower and hint in subf:
                    fused[key] = fused[key] * mult


        entity_sets = []
        for e in (primary if primary else entities):
            terms = _alts_for(e, alt_map)
            if terms:
                entity_sets.append(terms)

        if entity_sets:
            for key, score in list(fused.items()):
                doc = key2doc.get(key)
                if not doc:
                    continue
                text = (doc.page_content or "").lower()
                meta = " ".join(str(v) for v in doc.metadata.values()).lower()
                blob = text + " " + meta
                if all(_contains_any(blob, terms) for terms in entity_sets):
                    fused[key] = fused[key] + CO_MENTION_BONUS
                    provenance.setdefault(key, set()).add("co_mention")

        # ----------------------------
        # rank and coverage
        # ----------------------------
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        ranked_docs = [(key2doc[k], s, sorted(list(provenance.get(k, [])))) for k, s in ranked]

        top_k = min(self.pool_cap, len(ranked_docs))
        top = ranked_docs[:top_k]
        tail = ranked_docs[top_k:]

        # ensure each entity appears at least once in top
        missing = []
        for e in (primary if primary else entities):
            if e.strip() and not any(_doc_mentions_entity(d, e, alt_map) for d, _, _ in top):
                missing.append(e)

        if missing:
            for e in missing:
                promoted = False
                for idx, (d, s, src) in enumerate(tail):
                    if _doc_mentions_entity(d, e, alt_map):
                        # replace worst in top (last) with this doc
                        if top:
                            top.pop()
                        top.append((d, s, src + [f"coverage:{e}"]))
                        tail.pop(idx)
                        promoted = True
                        break
                # if still not found, we just keep the current top as-is

        # re-sort top after potential swaps
        top.sort(key=lambda x: x[1], reverse=True)

        # ----------------------------
        # printing diagnostics
        # ----------------------------
        dt = time.time() - t0
        print(f"\n2. Retrieval Statistics:")
        print(f"   Raw hits across lists: {total_raw_hits}")
        print(f"   Unique candidates: {len(uniq_keys)}")
        print(f"   Candidate pool size: {len(top)}")
        print(f"   Retrieval time: {dt:.3f} seconds")

        if top:
            scores = np.array([s for _, s, _ in top], dtype=float)
            print(f"\n3. Fused Score Distribution:")
            print(f"   Range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"   Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

            if len(scores) >= 2:
                gap = scores[0] - scores[1]
                print(f"\n4. Gap Analysis (Top-1 vs Top-2):")
                print(f"   Top-1: {scores[0]:.4f}, Top-2: {scores[1]:.4f}")
                print(f"   Gap: {gap:.4f}")
                if gap > 0.2:
                    print(f"   → Strong single target (gap > 0.2)")
                elif gap > 0.1:
                    print(f"   → Clear preference (0.1 < gap ≤ 0.2)")
                else:
                    print(f"   → Multiple relevant docs (gap ≤ 0.1)")

            N = min(TOP_PRINT, len(top))
            print(f"\n5. Top {N} Results (after fusion):")
            for i, (doc, score, src) in enumerate(top[:N], start=1):
                name = doc.metadata.get('name', 'Unknown')
                doc_type = doc.metadata.get('type', 'Unknown')
                src_str = ",".join(src)
                print(f"   {i}. [{doc_type}] {name}: fused={score:.4f}  from=[{src_str}]")

        # ----------------------------
        # return shape expected by Step 2
        # ----------------------------
        return [(doc, float(score)) for (doc, score, _) in top]

    def step2_rerank(
            self,
            candidates: List[Tuple[Document, float]],
            query: str,
            use_mmr: bool = True,
            mmr_lambda: float = 0.5,
            top_k: int = 12,
            temperature: float = 1.6,
            entity_prior: float = 0.6,  # post-softmax multiplicative prior for anchor docs
    ) -> List[Tuple[Document, float, float]]:
        """
        Step 2: Cross-encoder re-ranking with anchor-aware prior and MMR diversification.

        Returns:
            List of (Document, adj_raw_logit, prob) sorted by final probability desc.
            The 3rd element ('prob') is what Step-3 should use for relevance mass.
        """
        import numpy as np
        from sentence_transformers import CrossEncoder

        def _softmax_with_temp(x: np.ndarray, tau: float) -> np.ndarray:
            z = (x - x.max()) / max(tau, 1e-6)
            e = np.exp(z)
            s = e.sum()
            return e / (s if s > 0 else 1.0)

        def _norm_txt(s: str) -> str:
            return re.sub(r"[^\w\s]", " ", s.lower()).split()

        def _norm_key(s: str) -> str:
            # normalized phrase key
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower()).strip())

        def _parse_alt_names(meta_val) -> List[str]:

            out = []
            if not meta_val:
                return out
            if isinstance(meta_val, list):
                return [str(x).strip() for x in meta_val if str(x).strip()]
            s = str(meta_val)
            # remove brackets and stray backslashes/quotes
            s = s.strip()
            s = s.strip("[]")
            s = s.replace("\\", "")
            # split by comma
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return [p for p in parts if p]

        # ---------- 1) Prepare ----------
        if not candidates:
            return []

        # Load CE model once
        if not hasattr(self, "cross_encoder"):
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Build anchor set from the query (robust extraction)
        anchors = self._extract_query_anchors(query)

        # Map candidate docs to name/alt-name keys for anchor matching
        docs = [d for d, _ in candidates]
        step1_scores = [float(s) for _, s in candidates]

        name_keys_list = []
        is_anchor_doc = []
        for doc in docs:
            name = str(doc.metadata.get("name", "")).strip()
            alts = _parse_alt_names(doc.metadata.get("alternative_names", ""))
            keys = {_norm_key(name)} | {_norm_key(a) for a in alts if a}
            name_keys_list.append(keys)
            is_anchor_doc.append(any(_norm_key(a) in keys for a in anchors))

        # ---------- 2) Cross-encoder scoring ----------
        pairs = [[query, doc.page_content] for doc in docs]
        ce_raw = np.array(self.cross_encoder.predict(pairs), dtype=float)

        adj_raw = ce_raw.copy()

        # Boost scores for detected entities
        entities = self._extract_entities_simple(query)
        entity_names_lower = [e.lower() for e in entities]
        for i, doc in enumerate(docs):
            doc_name = doc.metadata.get("name", "").lower()
            if any(entity in doc_name for entity in entity_names_lower):
                adj_raw[i] += 2.0  # Small boost for entity matches

        # ---------- 3) Convert to probabilities (softmax with temperature) ----------
        probs = _softmax_with_temp(adj_raw, temperature)

        # ---------- 4) Apply post-softmax anchor prior & renormalize ----------
        if anchors:
            boost = np.ones_like(probs)
            for i, is_anchor in enumerate(is_anchor_doc):
                if is_anchor:
                    boost[i] *= (1.0 + max(0.0, entity_prior))
            probs = probs * boost
            probs = probs / probs.sum() if probs.sum() > 0 else probs

        # ---------- 5) Sort by probability ----------
        order = np.argsort(-probs)
        reranked = [(docs[i], float(adj_raw[i]), float(probs[i])) for i in order]

        for (doc, raw, prob) in reranked:
            if not isinstance(getattr(doc, "metadata", None), dict):
                doc.metadata = {}
            doc.metadata["ce_prob"] = float(prob)

        # ---------- 6) Ensure anchors are in the MMR candidate subset ----------
        if use_mmr and top_k > 1:
            subset = reranked[:top_k]
            # pull in any anchor docs that fell below top_k by replacing the tail
            tail_start = max(1, top_k - 3)  # keep top few intact
            tail = subset[tail_start:]

            # indices of anchor docs not already in subset
            subset_ids = {id(x[0]) for x in subset}
            anchor_pool = [t for t in reranked[top_k:] if any(
                _norm_key(a) in name_keys_list[docs.index(t[0])] for a in anchors
            )]
            changed = False
            for a_doc in anchor_pool:
                if id(a_doc[0]) not in subset_ids and tail:
                    # replace the last element in tail with anchor doc (if better for coverage)
                    tail[-1] = a_doc
                    changed = True

            if changed:
                subset = reranked[:tail_start] + tail
                # re-sort subset by prob desc
                subset.sort(key=lambda x: x[2], reverse=True)

            # ---------- 7) MMR diversification on subset ----------
            subset = self._apply_semantic_mmr(subset, query, mmr_lambda, top_k=len(subset))
            reranked = subset + reranked[top_k:]

        print(f"\n{'=' * 60}\nSTEP 2: CROSS-ENCODER RE-RANKING\n{'=' * 60}\n")
        print("Initializing cross-encoder model...")
        print("Cross-encoder loaded: ms-marco-MiniLM-L-6-v2\n")
        print(f"{'Rank':<4} {'Document':<30} {'Step1':>8}  {'CE Raw':>9}  {'Adj Raw':>9}  {'CE Prob':>8}  {'Ent?':>4}")
        print("-" * 79)
        for rank, (doc, raw, p) in enumerate(reranked[:5], 1):
            name = doc.metadata.get("name", "Unknown")[:28]
            ent = "*" if any(_norm_key(a) in name_keys_list[docs.index(doc)] for a in anchors) else ""
            s1 = step1_scores[docs.index(doc)] if doc in docs else 0.0
            print(f"{rank:<4} {name:<30} {s1:8.4f}  {raw:9.4f}  {raw:9.4f}  {p:8.4f}  {ent:>4}")

        print(f"\nApplying semantic MMR (lambda={mmr_lambda}, top_k={top_k})...")
        print("Semantic MMR applied")
        print(f"\nFinal statistics:\n  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"  Mean: {probs.mean():.4f}, Std: {probs.std():.4f}")

        return reranked

    def step3_adaptive_selection(
            self,
            reranked: List[Tuple[Document, float, float]],
            query: str,
            relevance_threshold: float = 0.85,
            max_tokens: int = 2000,
            min_docs: int = 1,
            max_docs_simple: int = 2,
            max_docs_complex: int = 5,
            anchor_min_prob: float = 0.008,  # tiny floor to ignore near-zeros
    ) -> List[Document]:
        """
        Step 3:  adaptive selection.
          1) Detect anchors.
          2) Pick the best doc for each anchor (by CE prob), if present and above floor.
          3) Fill remaining mass by descending prob (MMR order preserved), under token + max_docs caps.
        """
        import numpy as np

        def _norm_key(s: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower()).strip())

        def _parse_alt_names(meta_val) -> List[str]:
            out = []
            if not meta_val:
                return out
            if isinstance(meta_val, list):
                return [str(x).strip() for x in meta_val if str(x).strip()]
            s = str(meta_val)
            s = s.strip().strip("[]").replace("\\", "")
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return [p for p in parts if p]

        def _doc_token_len(doc: Document) -> int:
            # crude token estimate
            return max(1, len(doc.page_content) // 4)

        print(f"\n{'=' * 60}\nSTEP 3: ADAPTIVE DOCUMENT SELECTION\n{'=' * 60}\n")

        if not reranked:
            return []

        # Detect complexity
        ql = query.lower()

        # 1) Using simple entity extraction (same as Step 1)
        anchors = self._extract_entities_simple(query)
        anchors = [a.lower() for a in anchors]  # normalize to lowercase

        is_complex = any(
            w in ql for w in
            ["how", "why", "compare", "relation", "between", "explain", "difference", "versus", "vs"]) or len(
            anchors) > 2
        max_docs = max_docs_complex if is_complex else max_docs_simple

        # 2) Build name/alt sets for candidate docs
        name_sets = []
        for doc, raw, prob in reranked:
            name = str(doc.metadata.get("name", "")).strip()
            alts = _parse_alt_names(doc.metadata.get("alternative_names", ""))
            keys = {_norm_key(name)} | {_norm_key(a) for a in alts if a}
            name_sets.append(keys)

        # 3) Anchor coverage pass (select one best per anchor)
        selected_idx: List[int] = []
        covered: Set[str] = set()
        total_tokens = 0

        print("Query analysis:")
        print(f"  Query: '{query}'")
        print(f"  Type: {'Complex/Multi-aspect' if is_complex else 'Simple/Direct'}")
        print(f"  Target entities: {anchors if anchors else '(none detected)'}")

        # --- OOD / no-evidence abstain gate (MINIMAL ADD) ---
        # Uses CE probabilities already in `reranked` tuples: (doc, raw, prob)
        probs = [float(p) for _, _, p in reranked]
        probs_sorted = sorted(probs, reverse=True)
        top1 = probs_sorted[0] if probs_sorted else 0.0
        top2 = probs_sorted[1] if len(probs_sorted) > 1 else 0.0
        gap = (top1 - top2) if probs_sorted else 0.0

        K_LOOK = 5  # inspect first K items for "mass"
        MASS_TAU = 0.50  # tune 0.40–0.70
        TOP1_MIN = 0.15  # tune 0.10–0.25
        GAP_MIN = 0.03  # tune 0.02–0.08

        mass = sum(probs_sorted[:K_LOOK])

        # If scores are flat/low, abstain: return []
        if (top1 < TOP1_MIN) and (mass < MASS_TAU) and (gap < GAP_MIN):
            print(f"  Abstaining: low-confidence distribution "
                  f"(top1={top1:.3f}, mass@{K_LOOK}={mass:.3f}, gap={gap:.3f})")
            print("\nSelection summary:")
            print(f"  Documents selected: 0/{len(reranked)}")
            print(f"  Cumulative relevance (selected): 0.0%")
            print(f"  Estimated tokens: ~0")
            print(f"\n{'=' * 60}\nFINAL CONTEXT FOR LLM:\n{'=' * 60}\n")
            print(f"{'=' * 60}\nReady to send 0 documents to LLM\n{'=' * 60}")
            return []
        # --- end abstain gate ---

        for a in anchors:
            best_i, best_p = -1, -1.0
            for i, (doc, raw, prob) in enumerate(reranked):
                if _norm_key(a) in name_sets[i] and prob >= anchor_min_prob:
                    if prob > best_p:
                        best_p, best_i = prob, i
            if best_i >= 0:
                # check token budget & doc cap
                tok = _doc_token_len(reranked[best_i][0])
                if (len(selected_idx) < max_docs) and (total_tokens + tok <= max_tokens):
                    selected_idx.append(best_i)
                    covered.add(a)
                    total_tokens += tok
                    print(f"  Selected anchor doc: {reranked[best_i][0].metadata.get('name', 'Unknown')} "
                          f"(anchor='{a}', prob={best_p:.3f})")

        missing_anchors = set(anchors) - covered
        for missing in missing_anchors:
            # Find the best doc for this missing anchor
            for i, (doc, raw, prob) in enumerate(reranked):
                if i not in selected_idx:
                    doc_name = doc.metadata.get("name", "").lower()
                    if missing in doc_name:
                        tok = _doc_token_len(doc)
                        if (len(selected_idx) < max_docs) and (total_tokens + tok <= max_tokens):
                            selected_idx.append(i)
                            covered.add(missing)
                            total_tokens += tok
                            print(f"  Force-selected entity doc: {doc.metadata.get('name', 'Unknown')} "
                                  f"(entity='{missing}', prob={prob:.3f})")
                            break

        # 4) Fill to reach relevance mass with remaining docs by prob
        total_prob = float(sum(p for _, _, p in reranked))
        cum_prob = float(sum(reranked[i][2] for i in selected_idx)) / (total_prob if total_prob > 0 else 1.0)

        # iterate in reranked order (already diversified), skipping selected
        i = 0
        while (cum_prob < relevance_threshold or len(selected_idx) < min_docs) and len(
                selected_idx) < max_docs and i < len(reranked):
            if i not in selected_idx:
                doc, raw, prob = reranked[i]
                tok = _doc_token_len(doc)
                if prob >= anchor_min_prob and (total_tokens + tok) <= max_tokens:
                    selected_idx.append(i)
                    total_tokens += tok
                    cum_prob = float(sum(reranked[j][2] for j in selected_idx)) / (
                        total_prob if total_prob > 0 else 1.0)
                    # Reason for selection
                    if anchors and covered != set(anchors) and any(_norm_key(a) in name_sets[i] for a in anchors):
                        reason = "Anchor coverage"
                    else:
                        reason = f"Mass not met ({cum_prob:.2f} < {relevance_threshold:.2f})"
                    print(f"  Selected doc {i + 1}: {doc.metadata.get('name', 'Unknown')} - {reason}; prob={prob:.3f}")
            i += 1

        # 5) Minimum documents safeguard
        i = 0
        while len(selected_idx) < min_docs and len(selected_idx) < len(reranked):
            if i not in selected_idx:
                doc = reranked[i][0]
                tok = _doc_token_len(doc)
                if (total_tokens + tok) <= max_tokens and len(selected_idx) < max_docs:
                    selected_idx.append(i)
                    total_tokens += tok
                    print(f"  Added doc {i + 1}: {doc.metadata.get('name', 'Unknown')} - Minimum requirement")
            i += 1

        # 6) Coverage report
        missing = set(anchors) - covered if anchors else set()
        if anchors:
            print(
                f"\n  Entity coverage: {len(covered)}/{len(anchors)} ({(len(covered) / max(1, len(anchors))) * 100:.0f}%)")
            if missing:
                print(f"  Missing entities: {missing}")

        # 7) Summary & final print
        selected_idx_sorted = sorted(selected_idx, key=lambda k: -reranked[k][2])
        selected_docs = [reranked[i][0] for i in selected_idx_sorted]
        if selected_idx_sorted:
            last_cum = float(sum(reranked[j][2] for j in selected_idx_sorted)) / (total_prob if total_prob > 0 else 1.0)
        else:
            last_cum = 0.0

        print(f"\nSelection summary:")
        print(f"  Documents selected: {len(selected_docs)}/{len(reranked)}")
        print(f"  Cumulative relevance (selected): {last_cum * 100:.1f}%")
        print(f"  Estimated tokens: ~{total_tokens}")

        print(f"\n{'=' * 60}\nFINAL CONTEXT FOR LLM:\n{'=' * 60}\n")
        for rank, i in enumerate(selected_idx_sorted, 1):
            doc, raw, prob = reranked[i]
            name = doc.metadata.get("name", "Unknown")
            doc_type = doc.metadata.get("type", "Unknown")
            preview = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            print(f"--- Document {rank} ---")
            print(f"Type: {doc_type}")
            print(f"Name: {name} (prob={prob:.3f})")
            print(f"Content preview: {preview}\n")

        print(f"{'=' * 60}\nReady to send {len(selected_docs)} documents to LLM\n{'=' * 60}")

        return selected_docs

    '''
    def _extract_query_anchors(self, query: str) -> Set[str]:

        import re
        q = " " + query.lower() + " "
        # remove context phrases
        context_phrases = [
            "in the context of", "in terms of", "with respect to", "regarding",
            "related to", "as it relates to", "from the perspective of",
            "in relation to", "trade-offs between", "tradeoffs between",
            "trade-offs", "tradeoffs", "connection between", "relation between",
            "relationship between",
        ]
        for ph in context_phrases:
            q = q.replace(" " + ph + " ", " ")

        # quoted spans
        anchors = set(re.findall(r'"([^"]+)"', q))

        # between A and B
        for m in re.finditer(r"between\s+(.+?)\s+and\s+(.+?)(?:[?.!,;]|$)", q):
            anchors.add(m.group(1).strip())
            anchors.add(m.group(2).strip())

        # vs / versus
        for m in re.finditer(r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!,;]|$)", q):
            anchors.add(m.group(1).strip())
            anchors.add(m.group(2).strip())

        # split on commas/and/or and pick capitalized tokens from original query as hints
        # but allow single words (accuracy, completeness, age, adaptability, …)
        tokens = [t.strip() for t in re.split(r"[,/]| and | or ", q)]
        for t in tokens:
            tt = re.sub(r"[^\w\s]", " ", t).strip()
            # keep short KPI-like words too (e.g., 'age')
            if 2 <= len(tt) <= 48:
                anchors.add(tt)

        # normalize
        def _norm_key(s: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower()).strip())

        anchors = {_norm_key(a) for a in anchors if a}
        # drop generic words
        blacklist = {"how", "why", "what", "which", "explain", "compare", "vs", "versus", "relation", "relationship",
                     "difference", "trade", "trade offs", "tradeoffs", "cost", "quality", "context", "dataset",
                     "datasets"}
        anchors = {a for a in anchors if a not in blacklist}
        return anchors
    '''

    def _extract_query_anchors(self, query: str) -> Set[str]:
        """
        Robust anchor/keyword extractor that properly identifies domain entities.

        Extraction priority:
        1. Quoted phrases (highest confidence)
        2. Capitalized words/phrases (likely domain entities)
        3. Entities from structural patterns (between X and Y, X vs Y)
        4. Known domain entities (if available)
        5. Significant noun phrases

        Returns:
            Set of lowercase, normalized entity strings
        """
        import re
        anchors = set()
        original_query = query  # Preserve original for capitalization detection

        # --- 1. Extract quoted phrases (highest priority) ---
        quoted = re.findall(r'"([^"]+)"', query)
        for q in quoted:
            if q.strip():
                anchors.add(q.strip().lower())

        # --- 2. Extract capitalized words/phrases (domain entities) ---
        # This pattern captures:
        # - Single capitalized words: "Accuracy"
        # - Multi-word capitalized phrases: "Access Cost", "Data Quality"
        # - Mixed case phrases: "Return on Investment"
        capitalized_pattern = r'\b[A-Z][a-z]*(?:\s+(?:of|on|in|for|and|to|the)?\s*[A-Z]?[a-z]*)*\b'
        capitalized_matches = re.findall(capitalized_pattern, original_query)

        for match in capitalized_matches:
            # Clean and validate
            cleaned = match.strip()
            # Filter out single common words
            if cleaned.lower() not in {'explain', 'how', 'what', 'when', 'where',
                                       'why', 'the', 'in', 'of', 'for', 'and', 'or'}:
                if len(cleaned) > 1:  # Skip single letters
                    anchors.add(cleaned.lower())

        # --- 3. Extract from structural patterns ---
        query_lower = query.lower()

        # Pattern: "between X and Y"
        between_pattern = r'between\s+([^.?,;!]+?)\s+and\s+([^.?,;!]+?)(?:\s+(?:in|for|when|while|during)|[.?,;!]|$)'
        for match in re.finditer(between_pattern, query_lower):
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()

            # Clean entities - remove common prefixes/suffixes
            for entity in [entity1, entity2]:
                # Remove articles and common prefixes
                entity = re.sub(r'^(the|a|an)\s+', '', entity)
                # Remove trailing context words
                entity = re.sub(r'\s+(in|for|when|while|during|with).*$', '', entity)
                entity = entity.strip()

                if entity and len(entity) > 1:
                    anchors.add(entity)

        # Pattern: "trade-off(s) between X and Y"
        tradeoff_pattern = r'trade[\s-]?off?s?\s+between\s+([^.?,;!]+?)\s+and\s+([^.?,;!]+?)(?:\s|[.?,;!]|$)'
        for match in re.finditer(tradeoff_pattern, query_lower):
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()

            for entity in [entity1, entity2]:
                entity = re.sub(r'^(the|a|an)\s+', '', entity)
                entity = entity.strip()
                if entity and len(entity) > 1:
                    anchors.add(entity)

        # Pattern: "X vs/versus Y" or "X vs. Y"
        vs_pattern = r'\b([^.?,;!\s]+(?:\s+[^.?,;!\s]+)*)\s+(?:vs\.?|versus)\s+([^.?,;!\s]+(?:\s+[^.?,;!\s]+)*)'
        for match in re.finditer(vs_pattern, query_lower):
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()

            for entity in [entity1, entity2]:
                if entity and len(entity) > 1:
                    anchors.add(entity)

        # Pattern: "X or Y" in context of comparison/choice
        comparison_keywords = ['compare', 'difference', 'choose', 'decision', 'trade-off', 'versus', 'between']
        if any(kw in query_lower for kw in comparison_keywords):
            or_pattern = r'\b([^.?,;!\s]+(?:\s+[^.?,;!\s]+)*)\s+or\s+([^.?,;!\s]+(?:\s+[^.?,;!\s]+)*)'
            for match in re.finditer(or_pattern, query_lower):
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()

                for entity in [entity1, entity2]:
                    # Only add if it's a potential entity (capitalized in original or short phrase)
                    if entity and len(entity) > 1 and len(entity.split()) <= 3:
                        # Check if it was capitalized in original
                        if any(word[0].isupper() for word in entity.split() if word):
                            anchors.add(entity)

        # --- 4. Check against known entities (if available) ---
        # This helps catch entities even if they're not capitalized in the query
        if hasattr(self, 'known_entities'):
            query_words_lower = set(query_lower.split())

            for known_entity in self.known_entities:
                entity_lower = known_entity.lower()
                entity_words = entity_lower.split()

                # Check for exact match (all words present in sequence)
                if len(entity_words) == 1:
                    if entity_lower in query_words_lower:
                        anchors.add(entity_lower)
                else:
                    # Multi-word entity - check if it appears as a phrase
                    if entity_lower in query_lower:
                        anchors.add(entity_lower)

        # --- 5. Extract potential noun phrases (fallback) ---
        # Only if we haven't found enough entities yet
        if len(anchors) < 2:
            # Simple noun phrase pattern - words that might be entities
            # Look for: adjective* noun+ patterns
            noun_phrase_pattern = r'\b(?:[a-z]+\s+)?([a-z]+(?:\s+[a-z]+){0,2})\b'

            # Focus on phrases near key indicator words
            indicators = ['quality', 'cost', 'value', 'performance', 'efficiency',
                          'accuracy', 'speed', 'reliability', 'availability']

            for indicator in indicators:
                if indicator in query_lower:
                    # Get surrounding context
                    context_pattern = rf'(\w+(?:\s+\w+)?)\s+{indicator}|{indicator}\s+(\w+(?:\s+\w+)?)'
                    for match in re.finditer(context_pattern, query_lower):
                        for group in match.groups():
                            if group and len(group) > 1:
                                # Clean and validate
                                cleaned = group.strip()
                                if cleaned not in {'the', 'a', 'an', 'and', 'or', 'in', 'of', 'for'}:
                                    anchors.add(cleaned)

        # --- 6. Normalize and clean ---
        normalized_anchors = set()
        for anchor in anchors:
            # Normalize: lowercase, remove extra spaces, strip punctuation
            cleaned = re.sub(r'\s+', ' ', anchor.lower().strip())
            cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned)  # Strip leading/trailing non-word chars

            if cleaned:
                normalized_anchors.add(cleaned)

        # --- 7. Filter out noise ---
        # Comprehensive blacklist of non-entity words
        blacklist = {
            # Question words
            'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
            # Verbs
            'explain', 'describe', 'define', 'tell', 'show', 'give', 'provide',
            'analyze', 'compare', 'evaluate', 'assess',
            # Connectors and articles
            'the', 'a', 'an', 'and', 'or', 'but', 'with', 'without', 'between',
            'in', 'on', 'at', 'to', 'for', 'of', 'from', 'by', 'about', 'into',
            # Common adjectives that aren't entities
            'good', 'bad', 'best', 'worst', 'better', 'worse', 'new', 'old',
            # Generic terms
            'terms', 'context', 'perspective', 'relation', 'relationship',
            # Single letters
            'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your',
            # Partial phrases that slipped through
            'vs', 'versus'
        }

        # Additional filter: remove phrases that are too generic or too long
        filtered_anchors = set()
        for anchor in normalized_anchors:
            words = anchor.split()

            # Skip if any word is in blacklist (unless it's part of a known entity)
            if len(words) == 1 and words[0] in blacklist:
                continue

            # Skip overly long phrases (likely sentence fragments)
            if len(words) > 4:
                continue

            # Skip if entire phrase is in blacklist
            if anchor in blacklist:
                continue

            filtered_anchors.add(anchor)

        # --- 8. Special handling for specific patterns in your domain ---
        # If the query mentions specific concepts, ensure they're captured
        domain_keywords = {
            'accuracy', 'adaptability', 'accessibility', 'access cost',
            'data quality', 'data monetization', 'completeness', 'consistency',
            'precision', 'reliability', 'scalability', 'performance',
            'cost', 'value', 'efficiency', 'roi', 'return on investment'
        }

        for keyword in domain_keywords:
            if keyword in query_lower:
                filtered_anchors.add(keyword)

        # --- Debug output (optional) ---
        if hasattr(self, 'debug') and self.debug:
            print(f"Query: {query}")
            print(f"Detected entities: {filtered_anchors}")

        return filtered_anchors

    def _apply_semantic_mmr(self,
                            reranked: List[Tuple[Document, float, float]],
                            query: str,
                            mmr_lambda: float,
                            top_k: int) -> List[Tuple[Document, float, float]]:
        if not reranked or top_k <= 1:
            return reranked

        if not hasattr(self, "_mmr_embedder"):
            self._mmr_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Only embed the top_k (speed)
        texts = [doc.page_content for doc, _, _ in reranked[:top_k]]
        doc_embs = self._mmr_embedder.encode(texts, normalize_embeddings=True)
        q_emb = self._mmr_embedder.encode([query], normalize_embeddings=True)[0]

        # Use CE norm as relevance (consistent with your Step‑3), but cosine also available:
        ce_norms = np.array([n for _, _, n in reranked[:top_k]])

        selected = [0]  # keep the best CE doc
        remaining = list(range(1, top_k))

        while remaining:
            best_i, best = None, -1e9
            for i in remaining:
                rel = float(ce_norms[i])
                red = float(np.max(doc_embs[i] @ doc_embs[selected].T))  # cosine redundancy
                mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * red
                if mmr > best:
                    best, best_i = mmr, i
            selected.append(best_i)
            remaining.remove(best_i)

        diversified = [reranked[i] for i in selected] + reranked[top_k:]
        return diversified

    def _extract_query_anchors(self, query: str) -> Set[str]:
        """
        Robust anchor/keyword extractor:
          - handles single-word KPIs (Accuracy, Age, Completeness, Adaptability, ...)
          - strips punctuation, commas
          - captures 'between A and B', 'A vs B'
          - removes common context phrases (e.g., 'in the context of', 'in terms of')
        Returns lowercase, punctuation-stripped phrases.
        """
        import re
        q = " " + query.lower() + " "
        # remove context phrases
        context_phrases = [
            "in the context of", "in terms of", "with respect to", "regarding",
            "related to", "as it relates to", "from the perspective of",
            "in relation to", "trade-offs between", "tradeoffs between",
            "trade-offs", "tradeoffs", "connection between", "relation between",
            "relationship between",
        ]
        for ph in context_phrases:
            q = q.replace(" " + ph + " ", " ")

        # quoted spans
        anchors = set(re.findall(r'"([^"]+)"', q))

        # between A and B
        for m in re.finditer(r"between\s+(.+?)\s+and\s+(.+?)(?:[?.!,;]|$)", q):
            anchors.add(m.group(1).strip())
            anchors.add(m.group(2).strip())

        # vs / versus
        for m in re.finditer(r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!,;]|$)", q):
            anchors.add(m.group(1).strip())
            anchors.add(m.group(2).strip())

        # split on commas/and/or and pick capitalized tokens from original query as hints
        # but allow single words (accuracy, completeness, age, adaptability, …)
        tokens = [t.strip() for t in re.split(r"[,/]| and | or ", q)]
        for t in tokens:
            tt = re.sub(r"[^\w\s]", " ", t).strip()
            # keep short KPI-like words too (e.g., 'age')
            if 2 <= len(tt) <= 48:
                anchors.add(tt)

        # normalize
        def _norm_key(s: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower()).strip())

        anchors = {_norm_key(a) for a in anchors if a}
        # drop generic words
        blacklist = {"how", "why", "what", "which", "explain", "compare", "vs", "versus", "relation", "relationship",
                     "difference", "trade", "trade offs", "tradeoffs", "cost", "quality", "context", "dataset",
                     "datasets"}
        anchors = {a for a in anchors if a not in blacklist}
        return anchors



    def _get_document_key(self, doc: Document) -> str:
        """
        Generate a stable unique key for document mapping.
        """
        doc_type = doc.metadata.get("type", "")
        doc_name = doc.metadata.get("name", "")

        if doc_type and doc_name:
            return f"{doc_type}::{doc_name}"

        # Stable hash fallback
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
        return f"content::{content_hash}"

    def _search_dense(self, q: str, k: int) -> List[Tuple[Document, float]]:
        """
        Helper: run dense search for q and return [(doc, similarity)] sorted desc.
        Uses similarity_search_with_relevance_scores if available; otherwise converts distance to similarity.
        """
        try:
            pairs = self.vectorstore.similarity_search_with_relevance_scores(q, k=k)
            # Ensure sorted desc by score
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs
        except (AttributeError, NotImplementedError):
            pairs = self.vectorstore.similarity_search_with_score(q, k=k)
            # Convert distance->similarity
            out = [(d, 1.0 - dist) for (d, dist) in pairs]
            out.sort(key=lambda x: x[1], reverse=True)
            return out

    def _rrf_fuse(self,
                  per_list_results: List[Tuple[str, float, List[Tuple[Document, float]]]],
                  rrf_k: int = 60,
                  entities: List[str] = None,
                  aliases: dict = None):
        """
        Weighted RRF fusion with small exact-match/entity boosts.
        Returns dict: key -> {"doc": Document, "score": fused_score, "contrib": {tag: partial_score, ...}}
        """
        if entities is None:
            entities = []
        if aliases is None:
            aliases = {}

        # Build alias lookup set (lowercased) for quick name checks
        alias_sets = {}
        for e in entities:
            terms = aliases.get(e, [e])
            alias_sets[e] = {t.lower() for t in terms}

        fused = {}  # key -> {doc, score, contrib}
        for tag, weight, results in per_list_results:
            for rank, (doc, _sim) in enumerate(results, start=1):
                key = self._get_document_key(doc)
                rrf = weight * (1.0 / (rrf_k + rank))  # reciprocal rank with weight

                if key not in fused:
                    fused[key] = {"doc": doc, "score": 0.0, "contrib": {}}

                fused[key]["score"] += rrf
                fused[key]["contrib"][tag] = fused[key]["contrib"].get(tag, 0.0) + rrf

        # Exact-name / entity-name boosts (small, just to surface obvious matches)
        NAME_BOOST = 0.35  # title contains any entity alias
        BOTH_ENTITIES_BOOST = 0.20  # title contains aliases of two different entities (if multi-entity query)

        for key, info in fused.items():
            doc = info["doc"]
            title = (doc.metadata.get("name") or "").lower()
            matched_entities = set()

            for e, aset in alias_sets.items():
                if any(term in title for term in aset):
                    matched_entities.add(e)

            if matched_entities:
                info["score"] += NAME_BOOST
                info["contrib"]["name_boost"] = info["contrib"].get("name_boost", 0.0) + NAME_BOOST

            if len(matched_entities) >= 2:
                info["score"] += BOTH_ENTITIES_BOOST
                info["contrib"]["both_names"] = info["contrib"].get("both_names", 0.0) + BOTH_ENTITIES_BOOST

        return fused
