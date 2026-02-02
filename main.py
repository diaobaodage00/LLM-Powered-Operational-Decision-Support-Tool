"""
Refinery LLM Agent MVP Demo (Low-carbon scheduling)

What it does:
1) Reset refinery RL environment (which internally builds a Gurobi model).
2) Summarize current operational state (inventory / price / demand / carbon_weight).
3) Ask an "LLM planner" for a structured action (9-dim vector).
4) Validate action safety/ranges.
5) Step environment for N periods and print key KPIs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from openai import OpenAI

import numpy as np

# ------------------------------------------------------------
# IMPORTANT:
# Please make sure your project has:
#   - refinery_rl421.py   (contains class RefineryEnv)
#   - g2p421_inventory.py (contains class RefineryInventoryModel)
# Or modify imports below accordingly.
# ------------------------------------------------------------
from refinery_rl421 import RefineryEnv  

# -----------------------------
# 1) Utilities: state summary
# -----------------------------

MAIN_PRODUCTS = ["W92", "W95", "JET", "W00", "EEN"] 
CONTROL_UNITS = ["FC1", "FC2", "HCU"] # Which units we control modes for


def extract_state(env: RefineryEnv) -> Dict[str, Any]:
    """
    Build a compact, LLM-friendly state.
    env observation vector:
      [inventory(16), prices(16), demand_min(16), demand_max(16), period(1)]
    """
    obs = env._get_observation()
    model = env.model

    num_products = len(model.Des)
    inv = obs[0:num_products]
    prices = obs[num_products:2 * num_products]
    dmin = obs[2 * num_products:3 * num_products]
    dmax = obs[3 * num_products:4 * num_products]
    period = int(obs[-1])

    # map product -> metrics
    prod_metrics = {}
    for i, p in enumerate(model.Des):
        prod_metrics[p] = {
            "inventory": float(inv[i]),
            "price": float(prices[i]),
            "demand_min": float(dmin[i]),
            "demand_max": float(dmax[i]),
        }

    # Focus on main products
    focus = {p: prod_metrics[p] for p in MAIN_PRODUCTS if p in prod_metrics}

    # crude cost (single-period in env)
    crude_cost = float(getattr(model, "MX1_cost_fluctuation", [0.0])[0])

    return {
        "period": period,
        "carbon_weight": float(env.carbon_weight),
        "crude_cost": crude_cost,
        "focus_products": focus,
        "control_units": CONTROL_UNITS,
        "action_space": {
            "dim": 9,
            "meaning": {
                "0": "FC1_mode (0=A, 1=B)",
                "1": "FC2_mode (0=A, 1=B)",
                "2": "HCU_mode (0=A, 1=B)",
                "3": "H6O split ratio (range [0.4,0.6])", 
                "4": "HCJ->JET split ratio (range [0,1])",
                "5": "R3S->PL1 split ratio (range [0,1])",
                "6": "IV2->FC1 split ratio (range [0,1])",
                "7": "C1L->GF1 split ratio (range [0,1])",
                "8": "HCO->FC1 split ratio (range [0,1])",
            },
        },
    }


# -----------------------------------
# 2) LLM planner (mock for MVP)
# -----------------------------------

@dataclass
class Plan:
    action: List[float]
    rationale: str


def mock_llm_planner(state: Dict[str, Any]) -> Plan:
    """
    A simple policy to imitate an LLM planner:
    - If gasoline (W92/W95) demand is high and inventory low -> choose higher-output modes (B) for FC units
    - If carbon_weight is high -> prefer lower-emission choice (example: set FC2 to A)
    - Adjust a key split ratio (R3S->PL1) based on a rough demand pressure heuristic

    This is intentionally simple, but it demonstrates:
      state -> reasoning -> structured action
    """
    cw = state["carbon_weight"]
    fp = state["focus_products"]

    def demand_pressure(p: str) -> float:
        if p not in fp:
            return 0.0
        inv = fp[p]["inventory"]
        dmax = fp[p]["demand_max"]
        # pressure higher when demand_max >> inventory (avoid division by zero)
        return float(dmax / max(inv, 1.0))

    gas_pressure = max(demand_pressure("W92"), demand_pressure("W95"))
    jet_pressure = demand_pressure("JET")

    # mode decisions (0=A, 1=B)
    # heuristics: B when pressure high; but if carbon_weight high, reduce aggressive modes
    fc1_mode = 1.0 if gas_pressure > 0.20 else 0.0
    fc2_mode = 1.0 if gas_pressure > 0.25 else 0.0
    hcu_mode = 1.0 if jet_pressure > 0.20 else 0.0

    if cw >= 0.3:
        # pretend "A" is lower-emission or lower throughput; keep at least one FC in A
        fc2_mode = 0.0

    # split ratios (keep within env ranges)
    h6o_split = 0.50  # within [0.4, 0.6]
    hcj_to_jet = 0.55 if jet_pressure > 0.22 else 0.45
    r3s_to_pl1 = float(np.clip(0.60 if gas_pressure > 0.22 else 0.40, 0.0, 1.0))

    iv2_to_fc1 = 0.50
    c1l_to_gf1 = 0.50
    hco_to_fc1 = 0.50

    action = [
        fc1_mode, fc2_mode, hcu_mode,
        h6o_split, hcj_to_jet, r3s_to_pl1,
        iv2_to_fc1, c1l_to_gf1, hco_to_fc1
    ]

    rationale = (
        f"gas_pressure={gas_pressure:.3f}, jet_pressure={jet_pressure:.3f}, "
        f"carbon_weight={cw:.2f}. "
        f"Set modes (FC1={int(fc1_mode)}, FC2={int(fc2_mode)}, HCU={int(hcu_mode)}) "
        f"and key split r3s_to_pl1={r3s_to_pl1:.2f}."
    )
    return Plan(action=action, rationale=rationale)

def openai_llm_planner(state: Dict[str, Any]) -> Plan:
    """
    Use OpenAI Responses API + function calling to get a strict, structured action.
    """
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set. Please export OPENAI_MODEL.")
    # ---- Define tool (function) schema ----
    tools = [
        {
            "type": "function",
            "function": {
                "name": "propose_refinery_action",
                "description": "Propose a safe refinery scheduling action (9-dim) under low-carbon objective.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "array",
                            "minItems": 9,
                            "maxItems": 9,
                            "items": {"type": "number"},
                            "description": (
                                "Length-9 action vector: "
                                "[FC1_mode, FC2_mode, HCU_mode, H6O_split(0.4-0.6), "
                                "HCJ_to_JET(0-1), R3S_to_PL1(0-1), IV2_to_FC1(0-1), "
                                "C1L_to_GF1(0-1), HCO_to_FC1(0-1)]"
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Short reasoning: bottlenecks, inventory/demand pressure, carbon trade-off."
                        }
                    },
                    "required": ["action", "rationale"],
                    "additionalProperties": False
                }
            }
        }
    ]

    # ---- Build prompt (keep it compact & operational) ----
    system = (
        "You are an operations decision assistant for a refinery scheduling system.\n"
        "You must return a tool call to propose_refinery_action.\n"
        "Be conservative: if uncertain, choose balanced split ratios near 0.5 and avoid extreme mode changes.\n"
        "Never output free-form text; only call the function."
    )

    user = (
        "Given the current refinery state, propose the next-period action.\n\n"
        f"STATE_JSON:\n{json.dumps(state, ensure_ascii=False)}\n\n"
        "Constraints:\n"
        "- action[0..2] are modes in [0,1] (0=A, 1=B)\n"
        "- action[3] H6O_split must be in [0.4, 0.6]\n"
        "- action[4..8] split ratios must be in [0,1]\n"
        "Objective: maximize profit while minimizing carbon emissions, weighted by carbon_weight.\n"
        "Return a single propose_refinery_action tool call."
    )

    # ---- Call Responses API forcing tool usage ----
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "propose_refinery_action"}},
    )

    # ---- Extract tool arguments ----
    tool_calls = [o for o in resp.output if o.type == "function_call"]
    if not tool_calls:
        raise RuntimeError("Model did not return a function_call. Try a different model or prompt.")

    args = tool_calls[0].arguments
    if isinstance(args, str):
        args = json.loads(args)

    action = args.get("action", None)
    rationale = args.get("rationale", "")

    if not isinstance(action, list):
        raise ValueError(f"Invalid action type from model: {type(action)}")

    return Plan(action=action, rationale=rationale)


# -----------------------------------
# 3) Safety validation
# -----------------------------------

def validate_action(action: List[float]) -> Tuple[bool, str]:
    if not isinstance(action, list) or len(action) != 9:
        return False, "Action must be a list of length 9."

    # mode bits in [0,1]
    for i in range(3):
        if not (0.0 <= action[i] <= 1.0):
            return False, f"Mode action[{i}] out of range [0,1]."

    # h6o split in [0.4, 0.6] according to your env bounds
    if not (0.4 <= action[3] <= 0.6):
        return False, "action[3] (H6O split) must be in [0.4, 0.6]."

    # other ratios in [0,1]
    for i in range(4, 9):
        if not (0.0 <= action[i] <= 1.0):
            return False, f"Split ratio action[{i}] out of range [0,1]."

    return True, "ok"


# -----------------------------------
# 4) Run
# -----------------------------------

def run_episode(env: RefineryEnv, horizon: int, use_mock: bool, seed: int) -> None:
    obs, info = env.reset(seed=seed)
    print(f"[Reset] period={info.get('current_period')} carbon_weight={env.carbon_weight:.2f}")

    for t in range(horizon):
        state = extract_state(env)

        if use_mock:
            plan = mock_llm_planner(state)
        else:
            try:
                plan = openai_llm_planner(state)
                ok, msg = validate_action(plan.action)
                if not ok:
                    raise ValueError(msg)
            except Exception as e:
                plan = mock_llm_planner(state)  # fallback
                print(f"[Warning] OpenAI planner failed, using mock. Error: {e}")
                
        ok, msg = validate_action(plan.action)
        if not ok:
            raise ValueError(f"Unsafe action from planner: {msg}\nAction={plan.action}\nRationale={plan.rationale}")

        action = np.array(plan.action, dtype=np.float32)

        obs, reward, terminated, truncated, step_info = env.step(action)

        # Pull KPIs from env history (your env records profit, carbon, weighted_profit, etc.)
        profit = env.history["profits"][-1] if env.history.get("profits") else None
        carbon = env.history["carbon_emissions"][-1] if env.history.get("carbon_emissions") else None
        wprofit = env.history["weighted_profits"][-1] if env.history.get("weighted_profits") else None

        print("\n" + "=" * 80)
        print(f"[Period {t+1}] Planner rationale: {plan.rationale}")
        print(f"[Period {t+1}] Action: {plan.action}")
        print(f"[Period {t+1}] Reward(norm): {reward:.6f} | Profit: {profit} | Carbon: {carbon} | Weighted: {wprofit}")
        print("=" * 80)

        done = terminated or truncated
        if done:
            print("[Done] episode finished.")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=3, help="planning horizon steps for demo")
    parser.add_argument("--carbon-weight", type=float, default=0.3, help="carbon emission weight")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mock", action="store_true", help="use mock planner (no API needed)")
    args = parser.parse_args()

    env = RefineryEnv(planning_horizon=args.horizon, render_mode=None, carbon_weight=args.carbon_weight)
    run_episode(env, horizon=args.horizon, use_mock=args.mock or True, seed=args.seed)


if __name__ == "__main__":
    main()
