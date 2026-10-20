"""RefineEvo 7de015d5 pool_interface.py: the bidirectional experience pool and its JSON backend.

The backend keeps the same records in memory; the search checkpoint saves them instead of memory.json.
Model and embedding calls go through the search, which logs them.
"""

import json
import uuid

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .utils import extract_json_block, messages, text


def judge_trajectory(trajectory):
    """
    Judge whether the operator was successful based on the generated trajectory information.
    """
    if not trajectory:
        return False

    offspring = (trajectory or {}).get("offspring") or {}
    objective = offspring.get("objective")
    if objective is None:
        return False

    parents = trajectory.get("parents") or []
    parent_objs = [
        p.get("objective")
        for p in parents
        if isinstance(p, dict) and p.get("objective") is not None
    ]

    # When no parent information exists, consider it successful as long as there's an objective value.
    if not parent_objs:
        return True

    try:
        objective_type = (trajectory.get("objective_type") or "min").lower()
    except AttributeError:
        objective_type = "min"

    try:
        objective_val = float(objective)
    except (TypeError, ValueError):
        return False

    try:
        parent_vals = [float(val) for val in parent_objs]
    except (TypeError, ValueError):
        return False

    tolerance = 1e-9
    if objective_type == "max":
        baseline = max(parent_vals)
        return objective_val > baseline + tolerance
    else:
        baseline = min(parent_vals)
        return objective_val < baseline - tolerance


def process_memories(memories):
    """
    Process retrieved experiences, separating successful and failed experiences
    """
    if not memories:
        return {
            'success_exp': "No successful experience retrieved.",
            'fail_exp': "No failed experience retrieved."
        }

    success_summaries = []
    fail_summaries = []

    for memory in memories:
        distilled_raw = memory.get("distilled_items")
        is_success = memory.get("is_success", False)

        if not distilled_raw:
            continue

        # Serialize experience content
        if isinstance(distilled_raw, (list, dict)):
            try:
                serialized = json.dumps(distilled_raw, ensure_ascii=False)
            except TypeError:
                serialized = str(distilled_raw)
        else:
            serialized = str(distilled_raw)

        serialized = serialized.strip()
        if not serialized:
            continue

        # Parse list-formatted experiences
        if serialized.startswith("[") and serialized.endswith("]"):
            try:
                parsed = json.loads(serialized)
                for item in parsed:
                    if isinstance(item, (dict, list)):
                        item_str = json.dumps(item, ensure_ascii=False)
                    else:
                        item_str = str(item)

                    if is_success:
                        success_summaries.append(item_str)
                    else:
                        fail_summaries.append(item_str)
                continue
            except json.JSONDecodeError:
                pass

        # Add to corresponding list
        if is_success:
            success_summaries.append(serialized)
        else:
            fail_summaries.append(serialized)

    success_exp = "\n".join(success_summaries) if success_summaries else "No successful experience retrieved."
    fail_exp = "\n".join(fail_summaries) if fail_summaries else "No failed experience retrieved."

    return {
        'success_exp': success_exp,
        'fail_exp': fail_exp
    }


class ExperiencePool:
    """ExperiencePool over JSONMemoryBackend, with experience_filter="all" as in cfg/pool.yaml."""

    def __init__(self, ask, embed, problem_desc, func_desc, top_k=3):
        self._ask, self._embed = ask, embed
        self._system_prompt = text("system_reflector").format(problem_desc=problem_desc, func_desc=func_desc)
        self._top_k = top_k
        # JSONMemoryBackend.data
        self.data = []

    # ----- ExperiencePool -----

    def distill_trajectory(self, query, trajectory):
        """(is_success, distilled JSON text), or None when the reflector call fails."""
        is_success = judge_trajectory(trajectory)
        trajectory_json = json.dumps(trajectory, ensure_ascii=False, indent=2)
        user_prompt = text("user_reflector_success" if is_success else "user_reflector_fail").format(
            query=query, trajectory=trajectory_json)
        response = self._ask(messages(self._system_prompt, user_prompt), "reflection")
        if response is None:
            return None  # Upstream catches the exception and stores nothing.
        # Extract JSON content
        json_text = extract_json_block(response)
        if isinstance(json_text, str):
            json_text = json_text.strip()
        else:
            json_text = json.dumps(json_text, ensure_ascii=False)

        # Validate JSON format
        try:
            parsed_json = json.loads(json_text)
            # Ensure it's in list format
            if not isinstance(parsed_json, list):
                parsed_json = [parsed_json]
            # Re-serialize to ensure correct format
            json_text = json.dumps(parsed_json, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # JSON parsing failed, use fallback solution
            fallback = [{
                "summary": "Trajectory analysis unavailable due to invalid LLM output.",
                "recommendations": ["Review LLM output format", "Check system_reflector.txt prompt"],
                "applicable_when": "LLM parsing failure - please check the reflection prompt"
            }]
            json_text = json.dumps(fallback, ensure_ascii=False)

        return is_success, json_text

    def add_experience(self, query, trajectory, query_embedding=None):
        """
        Add a new experience to the memory repository; returns the stored record or None.
        `query_embedding` reuses the retrieval vector of the same query text.
        """
        if not query or not query.strip() or not trajectory:
            return None

        clean_query = query.strip()

        # Extract operator_goal from trajectory
        operator_goal = trajectory.get("operator_goal", "unknown")

        distilled = self.distill_trajectory(clean_query, trajectory)
        if distilled is None:
            return None
        is_success, distilled_memories = distilled

        # Parse vector representation
        if query_embedding is None:
            query_embedding = self._embed(clean_query)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        trajectory_payload = json.dumps(trajectory, ensure_ascii=False, default=str)

        # Prepare experience data for storage.
        experience_to_add = {
            "embedding": query_embedding,
            "metadata": {
                "query": clean_query,
                "trajectory": trajectory_payload,
                "distilled_items": distilled_memories,
                "is_success": is_success,
                "operator_goal": operator_goal,  # Record which operator generated this experience
                "n_retrieves": 0,  # Record the number of times this memory has been used
                "score": 0,  # Record the score of this memory
            },
            "document": clean_query,
        }

        self.add([experience_to_add])
        return experience_to_add

    def retrieve_memories(self, query, operator_goal):
        """
        Retrieve the top k memories for `query` among those of this operator goal.
        Returns (memories, query embedding); memories are snapshots of the stored metadata.
        """
        if not query or not query.strip():
            return [], None

        clean_query = query.strip()
        query_embedding = self._embed(clean_query)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        # Retrieve enough memories for filtering
        k = self._top_k
        initial_k = max(k * 10, 50)
        all_memories = self.query(query_embedding, initial_k)

        if not all_memories:
            return [], query_embedding

        # Filter memories by operator_goal
        filtered_memories = []
        for memory in all_memories:
            memory_operator_goal = memory.get("operator_goal", "unknown")
            if memory_operator_goal == operator_goal:
                filtered_memories.append(memory)

        # Take the top k memories (already sorted by similarity)
        selected_memories = filtered_memories[:k]

        # Update n_retrieves count for selected memories
        for memory in selected_memories:
            memory_id = memory.get("_id")
            if memory_id:
                current_n_retrieves = memory.get("n_retrieves", 0)
                self.update(memory_id, {"n_retrieves": current_n_retrieves + 1})

        return selected_memories, query_embedding

    def update_memory_score(self, memory, improved):
        """
        Score +1 if the memory led to improvement, else -1, from the retrieved snapshot's score.
        A negative score deletes the experience. Returns the new score.
        """
        if not memory:
            return None

        memory_id = memory.get("_id")
        if not memory_id:
            return None

        current_score = memory.get("score", 0)
        new_score = current_score + (1 if improved else -1)

        # If score becomes negative, delete this experience
        if new_score < 0:
            self.delete(memory_id)
        else:
            self.update(memory_id, {"score": new_score})
        return new_score

    # ----- JSONMemoryBackend -----

    def add(self, items):
        # Add unique ID to each item
        for item in items:
            if "_id" not in item:
                item["_id"] = str(uuid.uuid4())
        self.data.extend(items)

    def query(self, query_embedding, k):
        if not self.data:
            return []

        embeddings = np.array([item["embedding"] for item in self.data])
        query_embedding_np = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity(query_embedding_np, embeddings)[0]

        # Get the indices of the top k highest similarities.
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Return results containing _id and metadata
        return [{"_id": self.data[i]["_id"], **self.data[i]["metadata"]} for i in top_k_indices]

    def update(self, memory_id, updates):
        for item in self.data:
            if item.get("_id") == memory_id:
                # Update metadata
                item["metadata"].update(updates)
                return

    def delete(self, memory_id):
        for i, item in enumerate(self.data):
            if item.get("_id") == memory_id:
                del self.data[i]
                return
