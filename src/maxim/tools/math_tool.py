"""MathTool — agent-accessible mathematical cognition tool.

Routes between IPS (approximate, fast) and Angular Gyrus (exact, precise)
based on the operation type.  Registered in ToolRegistry so ExecAgent
can invoke math operations as tool calls.

Operations:
  Approximate (IPS):  compare, trend, anomaly, estimate_sum, estimate_mean, categorize
  Exact (AG):         compute, analyze
  Workspace:          store_value, recall_value
  Memory:             recall_method, recall_memory, list_memories, store_memory
"""

from __future__ import annotations

from typing import Any

import time
import uuid

from maxim.math.angular_gyrus import AngularGyrus
from maxim.math.ips import IPS
from maxim.math.math_types import MathMemory
from maxim.math.types import MathCategory
from maxim.math.workspace import NumericalWorkspace
from maxim.tools.base import Tool


class MathTool(Tool):
    """Mathematical cognition tool — routes between IPS and Angular Gyrus."""

    name = "math"
    description = (
        "Perform mathematical operations. "
        "Approximate (fast): compare, trend, anomaly, estimate_sum, estimate_mean, categorize, "
        "assess_randomness. "
        "Exact (precise): compute (add/subtract/multiply/divide/power/modulo), "
        "sqrt/square_root {value}, cube_root {value}, squared {value}, cubed {value}, factorial {value}, "
        "analyze (descriptive/linear/quadratic/percentiles), "
        "mat_multiply, eigenvalues, solve_system, determinant. "
        "Workspace: store_value, recall_value. "
        "Memory: recall_method, recall_memory (by category/domain/name), "
        "list_memories (browse available records), store_memory (persist learned knowledge)."
    )
    input_schema: dict[str, Any] = {
        "operation": str,
    }

    def __init__(
        self,
        ips: IPS,
        angular_gyrus: AngularGyrus,
        workspace: NumericalWorkspace,
    ) -> None:
        self._ips = ips
        self._ag = angular_gyrus
        self._workspace = workspace
        super().__init__()

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Route to IPS or Angular Gyrus based on operation type."""
        operation = kwargs.get("operation", "")

        # --- Operation aliases (natural phrasing → canonical form) ---
        op_lower = operation.lower().replace("_", " ").strip()
        if op_lower in ("square root", "sqrt"):
            value = float(kwargs.get("value", kwargs.get("a", 0)))
            kwargs = {**kwargs, "op_type": "power", "operands": [value, 0.5]}
            operation = "compute"
        elif op_lower in ("cube root", "cbrt"):
            value = float(kwargs.get("value", kwargs.get("a", 0)))
            kwargs = {**kwargs, "op_type": "power", "operands": [value, 1.0 / 3.0]}
            operation = "compute"
        elif op_lower == "squared":
            value = float(kwargs.get("value", kwargs.get("a", 0)))
            kwargs = {**kwargs, "op_type": "power", "operands": [value, 2]}
            operation = "compute"
        elif op_lower == "cubed":
            value = float(kwargs.get("value", kwargs.get("a", 0)))
            kwargs = {**kwargs, "op_type": "power", "operands": [value, 3]}
            operation = "compute"
        elif op_lower == "factorial":
            value = int(float(kwargs.get("value", kwargs.get("a", 0))))
            if value < 0 or value > 170:
                return {"error": "Factorial only supports integers 0-170"}
            result_val = 1
            for i in range(2, value + 1):
                result_val *= i
            return {
                "operation": "factorial",
                "system": "exact",
                "value": result_val,
                "verbal": f"{value}! = {result_val}",
                "code": f"math.factorial({value})",
            }

        # --- IPS (Approximate Number System) routes ---

        if operation == "compare":
            a = float(kwargs.get("a", 0))
            b = float(kwargs.get("b", 0))
            result, confidence = self._ips.compare_with_confidence(a, b)
            return {
                "operation": "compare",
                "system": "approximate",
                "result": result.name,
                "confidence": confidence,
                "a": a,
                "b": b,
            }

        if operation == "trend":
            data = kwargs.get("data", [])
            result = self._ips.detect_trend([float(v) for v in data])
            return {
                "operation": "trend",
                "system": "approximate",
                "direction": result.direction.name,
                "slope_estimate": result.slope_estimate,
                "confidence": result.confidence,
                "magnitude": result.magnitude.name,
            }

        if operation == "anomaly":
            value = float(kwargs.get("value", 0))
            history = [float(v) for v in kwargs.get("history", [])]
            result = self._ips.detect_anomaly(value, history)
            return {
                "operation": "anomaly",
                "system": "approximate",
                "is_anomalous": result.is_anomalous,
                "deviation": result.deviation,
                "direction": result.direction,
                "confidence": result.confidence,
                "magnitude": result.magnitude.name,
            }

        if operation == "estimate_sum":
            data = [float(v) for v in kwargs.get("data", [])]
            result = self._ips.estimate_sum(data)
            return {
                "operation": "estimate_sum",
                "system": "approximate",
                "value": result.value,
                "magnitude": result.magnitude.name,
                "confidence": result.confidence,
            }

        if operation == "estimate_mean":
            data = [float(v) for v in kwargs.get("data", [])]
            result = self._ips.estimate_mean(data)
            return {
                "operation": "estimate_mean",
                "system": "approximate",
                "value": result.value,
                "magnitude": result.magnitude.name,
                "confidence": result.confidence,
            }

        if operation == "categorize":
            value = float(kwargs.get("value", 0))
            category = self._ips.categorize(value)
            return {
                "operation": "categorize",
                "system": "approximate",
                "value": value,
                "magnitude": category.name,
            }

        # --- Angular Gyrus (Exact) routes ---

        if operation == "compute":
            op_type = kwargs.get("op_type", "add")
            operands = [float(v) for v in kwargs.get("operands", [])]
            result = self._ag.compute(op_type, operands)
            self._workspace.record_computation(
                f"compute_{op_type}",
                {"operands": operands},
                result,
            )
            return {
                "operation": "compute",
                "system": "exact",
                "op_type": op_type,
                "value": result.value,
                "verbal": result.verbal,
                "code": result.code,
            }

        if operation == "analyze":
            data = [float(v) for v in kwargs.get("data", [])]
            method = kwargs.get("method", "descriptive")
            result = self._ag.analyze(data, method)
            self._workspace.record_computation(
                f"analyze_{method}",
                {"data_len": len(data), "method": method},
                result,
            )
            return {
                "operation": "analyze",
                "system": "exact",
                "method": result.method,
                "parameters": result.parameters,
                "verbal": result.verbal,
                "code": result.code,
                "confidence": result.confidence,
            }

        # --- Workspace routes ---

        if operation == "store_value":
            name = kwargs.get("name", "")
            value = kwargs.get("value")
            if name and value is not None:
                if isinstance(value, list):
                    self._workspace.store(name, [float(v) for v in value], source="tool")
                else:
                    self._workspace.store(name, float(value), source="tool")
                return {"operation": "store_value", "name": name, "stored": True}
            return {"operation": "store_value", "error": "name and value required"}

        if operation == "recall_value":
            name = kwargs.get("name", "")
            nv = self._workspace.recall(name)
            if nv:
                return {
                    "operation": "recall_value",
                    "name": nv.name,
                    "value": nv.value,
                    "magnitude": nv.magnitude.name,
                    "source": nv.source,
                }
            return {"operation": "recall_value", "name": name, "found": False}

        # --- Memory routes ---

        if operation == "recall_method":
            description = kwargs.get("description", "")
            record = self._ag.recall_method(description)
            if record:
                return {
                    "operation": "recall_method",
                    "name": record.name,
                    "verbal": record.verbal,
                    "code": record.code,
                    "domain": record.domain,
                    "confidence": record.confidence,
                }
            return {"operation": "recall_method", "found": False}

        # --- IPS randomness assessment ---

        if operation == "assess_randomness":
            data = [float(v) for v in kwargs.get("data", [])]
            result = self._ips.assess_randomness(data)
            return {
                "operation": "assess_randomness",
                "system": "approximate",
                "is_random": result.is_random,
                "pattern_confidence": result.pattern_confidence,
                "pattern_type": result.pattern_type.name,
                "runs_z_score": result.runs_z_score,
                "autocorrelation": result.autocorrelation,
            }

        # --- Angular Gyrus matrix operations ---

        if operation == "mat_multiply":
            a = kwargs.get("a", [])
            b = kwargs.get("b", [])
            result = self._ag.mat_multiply(a, b)
            return {
                "operation": "mat_multiply",
                "system": "exact",
                "value": result.value,
                "verbal": result.verbal,
                "code": result.code,
            }

        if operation == "eigenvalues":
            matrix = kwargs.get("matrix", [])
            result = self._ag.mat_eigenvalues(matrix)
            return {
                "operation": "eigenvalues",
                "system": "exact",
                "value": result.value,
                "verbal": result.verbal,
                "code": result.code,
            }

        if operation == "solve_system":
            coefficients = kwargs.get("coefficients", [])
            constants = [float(v) for v in kwargs.get("constants", [])]
            result = self._ag.solve_system(coefficients, constants)
            return {
                "operation": "solve_system",
                "system": "exact",
                "value": result.value,
                "verbal": result.verbal,
                "code": result.code,
            }

        if operation == "determinant":
            matrix = kwargs.get("matrix", [])
            result = self._ag.mat_determinant(matrix)
            return {
                "operation": "determinant",
                "system": "exact",
                "value": result.value,
                "verbal": result.verbal,
                "code": result.code,
            }

        # --- Angular Gyrus persistent memory operations ---

        if operation == "recall_memory":
            name = kwargs.get("name")
            category = kwargs.get("category")
            domain = kwargs.get("domain")
            min_confidence = kwargs.get("min_confidence")
            limit = int(kwargs.get("limit", 5))
            if min_confidence is not None:
                min_confidence = float(min_confidence)
            records = self._ag.recall(
                limit=limit,
                name=name,
                category=category,
                domain=domain,
                min_confidence=min_confidence,
            )
            results = []
            for r in records:
                entry: dict[str, Any] = {
                    "id": r.id,
                    "name": r.name,
                    "category": r.category.name,
                    "domain": r.domain,
                    "confidence": r.confidence,
                }
                if hasattr(r, "verbal"):
                    entry["verbal"] = r.verbal
                    entry["code"] = r.code
                results.append(entry)
            return {
                "operation": "recall_memory",
                "count": len(results),
                "records": results,
            }

        if operation == "list_memories":
            category = kwargs.get("category")
            domain = kwargs.get("domain")
            limit = int(kwargs.get("limit", 20))
            records = self._ag.recall(
                limit=limit,
                category=category,
                domain=domain,
            )
            results = []
            for r in records:
                results.append({
                    "id": r.id,
                    "name": r.name,
                    "category": r.category.name,
                    "domain": r.domain,
                    "confidence": r.confidence,
                    "access_count": r.access_count,
                })
            return {
                "operation": "list_memories",
                "count": len(results),
                "records": results,
            }

        if operation == "store_memory":
            name = kwargs.get("name", "")
            category_str = kwargs.get("category", "PATTERN")
            domain = kwargs.get("domain", "learned")
            verbal = kwargs.get("verbal", "")
            code = kwargs.get("code", "")
            confidence = float(kwargs.get("confidence", 0.8))
            if not name:
                return {"operation": "store_memory", "error": "name required"}
            try:
                category = MathCategory[category_str.upper()]
            except (KeyError, AttributeError):
                category = MathCategory.PATTERN
            now = time.time()
            record = MathMemory(
                id=f"tool_{name}_{uuid.uuid4().hex[:8]}",
                timestamp=now,
                name=name,
                category=category,
                domain=domain,
                verbal=verbal,
                code=code,
                source="learned",
                confidence=confidence,
            )
            record_id = self._ag.store(record)
            return {
                "operation": "store_memory",
                "stored": True,
                "record_id": record_id,
                "name": name,
                "category": category.name,
            }

        return {"error": f"Unknown math operation: {operation}"}
