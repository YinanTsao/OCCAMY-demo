import random
import json
import os
from collections import defaultdict
from pprint import pprint  # <-- added for pretty printing
import math
MAX_RECURSION = 10



def check_request_load_feasibility(applications, placement, assignment, verbose=True):
    """
    Check if deployed model variants can handle assigned user request loads.
    Prints a detailed report if verbose=True.

    Returns:
        feasible (bool)
        detailed_report (dict)
    """

    report = {}
    feasible = True

    if verbose:
        print("\n=== REQUEST LOAD FEASIBILITY CHECK ===\n")

    for site, models in placement.items():
        for model, variants in models.items():
            for variant, replicas in variants.items():

                if replicas == 0:
                    continue

                # Service rate
                if variant == model:
                    service_rate = applications[model]["service_rate"]
                else:
                    service_rate = applications[model]["degrading_models"][variant]["service_rate"]

                max_load = service_rate * replicas

                # Incoming request load
                incoming_load = 0.0
                users_served = []
                for user, user_models in assignment.items():
                    if user_models.get(model) == site:
                        rate = applications[model]["request_rates"].get(user, 0)
                        incoming_load += rate
                        users_served.append((user, rate))

                is_feasible = incoming_load <= max_load
                feasible &= is_feasible

                report[(site, model, variant)] = {
                    "incoming_load": incoming_load,
                    "service_capacity": max_load,
                    "replicas": replicas,
                    "service_rate_per_replica": service_rate,
                    "users": users_served,
                    "feasible": is_feasible
                }

                if verbose:
                    status = "OK ✅" if is_feasible else "OVERLOADED ❌"
                    print(f"Site: {site} | Model: {model} | Variant: {variant}")
                    print(f"  Replicas            : {replicas}")
                    print(f"  Service rate        : {service_rate:.4f} req/s per replica")
                    print(f"  Total capacity      : {max_load:.4f} req/s")
                    print(f"  Incoming load       : {incoming_load:.4f} req/s")
                    print(f"  Users served        : {users_served}")
                    print(f"  Status              : {status}\n")

    if verbose:
        if feasible:
            print("✅ ALL LOAD CONSTRAINTS SATISFIED\n")
        else:
            print("❌ LOAD CONSTRAINTS VIOLATED\n")

    return feasible, report



def build_output_json(applications, placement, assignment,ouput_dir):
    """
    Convert placement and assignment results into structured JSON format.
    """
    output = {"sites": []}

    for site, models in placement.items():
        site_entry = {"site": site, "applications": []}

        for model_name, variants in models.items():
            for variant_name, instance_count in variants.items():
                # Gather users assigned to this model/variant at this site
                users_list = []
                for user, user_models in assignment.items():
                    if user_models.get(model_name) == site:
                        users_list.append({
                            "user": user,
                            "request_rate": applications[model_name]["request_rates"].get(user, 0)
                        })
                if users_list:
                    app_entry = {
                        "app": variant_name,
                        "instances": instance_count,
                        "users": users_list
                    }
                    site_entry["applications"].append(app_entry)

        output["sites"].append(site_entry)
        # Output file path
    output_path = os.path.join(ouput_dir, "output", "best_placement_MVSP.json")

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write JSON to file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nPlacement JSON saved to: {output_path}")

    return output

# -------------------------------
# Utilization cost
# -------------------------------
def compute_utilization_cost(placement, capacities):
    """
    Compute node utilization and exponential cost
    """
    node_costs = {}
    for site, models in placement.items():
        used_capacity = 0
        for model, variants in models.items():
            for variant, count in variants.items():
                slots = applications[model]["slots"] if variant == model else applications[model]["degrading_models"][variant]["slots"]
                used_capacity += slots * count
        u_e =  min(used_capacity / capacities[site], 1.0) # utilization
        cost = math.exp(u_e) - 1  # exponential cost function
        node_costs[site] = cost
    avg_cost = sum(node_costs.values()) / len(node_costs)
    return node_costs, avg_cost


# -------------------------------
# Helper: Compute latency
# -------------------------------
def compute_latency(user, site, model_name, variant_name, placement, alpha=0.1):
    """
    Total latency for a user accessing a model-variant on a site
    """
    # Communication latency
    CL = applications[model_name]["latency_site_user"][str((site, user))]
    L_max = 0
    # Base inference latency (service rate used as proxy for simplicity)
    if variant_name == model_name:
        IL_base = applications[model_name]["service_rate"]
        L_max = applications[model_name]["slo"]
    else:
        IL_base = applications[model_name]["degrading_models"][variant_name]["service_rate"]
        L_max = applications[model_name]["degrading_models"][variant_name]["slo"]

    # Replication count
    n_replicas = placement.get(site, {}).get(model_name, {}).get(variant_name, 0)

    # Colocation penalty
    colocation_penalty = 0
    for m, variants in placement.get(site, {}).items():
        for v, count in variants.items():
            if m != model_name or v != variant_name:
                sr = applications[m]["service_rate"] if v == m else applications[m]["degrading_models"][v]["service_rate"]
                colocation_penalty += alpha * sr * count

    # Total inference latency
    IL = IL_base + alpha * IL_base * max(n_replicas - 1, 0) + colocation_penalty

    return CL + IL, ( CL + IL)/L_max

# -------------------------------
# Random Placement Algorithm (RPA)
# -------------------------------
def place(model_name, sites, placement, capacities):
    variants = [model_name] + list(applications[model_name].get("degrading_models", {}).keys())
    random.shuffle(variants)
    for variant in variants:
        for site in random.sample(sites, len(sites)):
            slots_needed = applications[model_name]["slots"] if variant == model_name else applications[model_name]["degrading_models"][variant]["slots"]
            if capacities[site] >= slots_needed:
                placement.setdefault(site, {}).setdefault(model_name, {}).setdefault(variant, 0)
                placement[site][model_name][variant] += 1
                capacities[site] -= slots_needed
                return placement,capacities.copy()
    return placement,capacities

# -------------------------------
# Inference Assignment Algorithm (IAA)
# -------------------------------
def assign_with_no_scaling(user, model_name, placement,remaining_cap,assignment):
    """
    Assign user to a deployed variant that satisfies latency and load requirement
    """
    candidates = []
    for site, models in placement.items():
        for variant, count in models.get(model_name, {}).items():
            if count > 0:
                # Compute latency
                latency_val, norm_latency_val = compute_latency(user, site, model_name, variant, placement)
                # Compute current load for this variant
                assigned_users = [u for u, models in assignment.items() if models.get(model_name) == site]
                total_load = sum(applications[model_name]["request_rates"].get(u, 0) for u in assigned_users)
                max_load = applications[model_name]["service_rate"] if variant == model_name else applications[model_name]["degrading_models"][variant]["service_rate"]
                
                if latency_val <= applications[model_name]["slo"] and total_load < max_load:
                    candidates.append((site, variant, latency_val))

    # Pick the one with lowest latency
    if candidates:
        site, variant, _ = min(candidates, key=lambda x: x[2])
        return site, variant
    else:
        # No feasible assignment, attempt to place new variant
        placement,capacities = place(model_name, sites, placement, remaining_cap)
        # Retry assignment recursively
        remaining_cap = capacities.copy()
        return assign_with_no_scaling(user, model_name, placement,remaining_cap,assignment)


# -------------------------------
# Inference Assignment Algorithm (IAA) with scaling
# -------------------------------


def assign(user, model_name, placement, remaining_cap, assignment, depth=0, K=5):
    """
    Assign user to a deployed variant.
    Scales replicas if overloaded.
    Stops after MAX_RECURSION attempts.
    """

    if depth >= MAX_RECURSION:
        return None, None

    candidates = []

    for site, models in placement.items():
        for variant, replicas in models.get(model_name, {}).items():
            if replicas == 0:
                continue

            # Service characteristics
            if variant == model_name:
                service_rate = applications[model_name]["service_rate"]
                slo = applications[model_name]["slo"]
                slots = applications[model_name]["slots"]
            else:
                service_rate = applications[model_name]["degrading_models"][variant]["service_rate"]
                slo = applications[model_name]["degrading_models"][variant]["slo"]
                slots = applications[model_name]["degrading_models"][variant]["slots"]

            # Current load
            assigned_users = [
                u for u, models in assignment.items()
                if models.get(model_name) == site
            ]
            current_load = sum(applications[model_name]["request_rates"].get(u, 0)
                               for u in assigned_users)

            user_load = applications[model_name]["request_rates"].get(user, 0)
            max_load = service_rate * replicas

            # Latency
            latency_val, _ = compute_latency(user, site, model_name, variant, placement)

            # Case 1: fits directly
            if latency_val <= slo and current_load + user_load <= max_load:
                candidates.append((site, variant, latency_val))
                continue

            # Case 2: try replica scaling
            if replicas < K and remaining_cap[site] >= slots:
                placement[site][model_name][variant] += 1
                remaining_cap[site] -= slots
                new_max_load = service_rate * (replicas + 1)

                if current_load + user_load <= new_max_load:
                    candidates.append((site, variant, latency_val))

    # Select best candidate
    if candidates:
        site, variant, _ = min(candidates, key=lambda x: x[2])
        return site, variant

    # Try placing a new variant (recursive attempt)
    placement, capacities = place(model_name, sites, placement, remaining_cap)
    remaining_cap.update(capacities)

    return assign(user,model_name,placement,remaining_cap,assignment,depth=depth + 1)


# -------------------------------
# General Heuristic (GH)
# -------------------------------
def general_heuristic(applications, sites, pricing, capacities):
    placement = {}
    assignment = defaultdict(dict)
    user_model_variant = defaultdict(dict)
    remaining_cap = capacities.copy()

    for model_name in applications.keys():
        placement,new_capacities = place(model_name, sites, placement, remaining_cap)
        remaining_cap = new_capacities.copy()
        for user in applications[model_name]["users"]:
            site, variant = assign(user, model_name, placement, remaining_cap, assignment)
            if site is None:
                print(f"Could not assign user {user} for model {model_name}")
                continue
            assignment[user][model_name] = site
            user_model_variant[user][model_name] = variant if site else None


    total_cost = 0
    for site, models in placement.items():
        for model, variants in models.items():
            for variant, count in variants.items():
                slot_count = applications[model]["slots"] if variant == model else applications[model]["degrading_models"][variant]["slots"]
                total_cost += pricing[site] * slot_count * count

    total_latency = 0
    total_norm_latency = 0
    total_requests = 0
    for user, models in user_model_variant.items():
        for model_name, variant in models.items():
            if variant is not None:
                site = assignment[user][model_name]
                latency_val, norm_latency_val = compute_latency(user, site, model_name, variant, placement)
                total_latency += latency_val
                total_norm_latency += norm_latency_val
                total_requests += 1

    avg_latency = total_latency / max(total_requests, 1)
    avg_norm_latency = total_norm_latency / max(total_requests, 1)



    node_usage_costs, avg_node_usage_cost = compute_utilization_cost(placement, capacities)

    return placement, assignment, user_model_variant, total_cost, avg_latency, avg_norm_latency, node_usage_costs, avg_node_usage_cost


def objective(alpha, avg_latency, avg_node_usage_cost):
    return alpha * avg_latency + (1 - alpha) * avg_node_usage_cost


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":


   
    input_file = 'input/multi/#7.json'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print()
    json_file = os.path.join(script_dir, input_file)
    with open(json_file, 'r') as f:
        input_data = json.load(f)

    sites = input_data["sites"]
    capacities = input_data["capacities"]
    pricing = input_data["pricing"]
    applications = input_data["applications"]



    N_RUNS = 30
    ALPHA = 0.5   # equal latency-cost objective

    best_result = None
    best_P = float("inf")

    for run in range(N_RUNS):
        # IMPORTANT: reset capacities each run
        capacities_copy = capacities.copy()
        placement, assignment, user_model_variant, total_cost, avg_latency, avg_norm_latency, node_usage_costs, avg_node_usage_cost = general_heuristic(applications, sites, pricing, capacities_copy)


        P = objective(ALPHA, avg_latency, total_cost)

        print(f"Run {run+1:02d} | L={avg_latency:.2f}, C={total_cost:.3f}, P={P:.3f}")

        if P < best_P:
            best_P = P
            best_result = {
                "placement": placement,
                "assignment": assignment,
                "user_model_variant": user_model_variant,
                "total_cost": total_cost,
                "avg_latency": avg_latency,
                "node_usage_costs": node_usage_costs,
                "avg_node_usage_cost": avg_node_usage_cost,
                "P": P
            }


    print("\n================ BEST SOLUTION ================\n")
    print(f"Objective P = {best_result['P']:.3f}")
    print(f"Average Latency L = {best_result['avg_latency']:.2f}")
    print(f"Average Utilization Cost C = {best_result['avg_node_usage_cost']:.3f}")
    check_request_load_feasibility(applications, best_result["placement"], best_result["assignment"], verbose=True)
    output_json = build_output_json(
        applications,
        best_result["placement"],
        best_result["assignment"],
        script_dir
    )


