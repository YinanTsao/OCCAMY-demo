import argparse
import math
from gurobipy import Model, GRB
import scipy.optimize as spo
import os
import sys
import json
import time
import itertools
import random
from decimal import Decimal, getcontext


# Queueing time distribution function
def waitingTimeDistr(t, lm, service_rate): 
    mu = service_rate
    if lm >= mu:
        return 0.0
    if lm <= 0:
        return 1.0
    
    getcontext().prec = 80
    s = Decimal(0)
    dlm = Decimal(lm)
    dmu = Decimal(mu)
    dt = Decimal(t)
    for i in range(int(t * mu) + 1):
        di = Decimal(i)
        t1 = di / dmu - dt
        f1 = Decimal.exp(-dlm * t1)
        f2 = t1 ** di
        f3 = dlm ** di
        f4 = math.factorial(i)
        s = s + f1 * f2 * f3 / f4    
    return float((Decimal(1) - dlm / dmu) * s)



def initialize_model(sites, pricing, opti_pref, applications, capacities, theta):
    # Initialize the model
    model = Model("Cerberus_Gopt")

    # Set Gurobi parameters
    # model.setParam(GRB.Param.TimeLimit, 2000)  # Limit the solver to 60 seconds
    # model.setParam(GRB.Param.MIPFocus, 1)    # Focus on finding feasible solutions faster
    # model.setParam(GRB.Param.Cuts, 3)        # Use aggressive cutting planes
    # model.setParam(GRB.Param.Presolve, 1)    # Enable strong presolve to reduce problem size
    # model.setParam(GRB.Param.Heuristics, 0.9) # Increase heuristic search

    # Decision variables
    x = {} # Assignment of users to sites
    y = {} # Open sites
    y_site = model.addVars(sites, vtype=GRB.BINARY, name="y_site")
    # y = model.addVars(sites, vtype=GRB.BINARY, name=f"y")  # Moving it to global constrant to make the min_site available
    u = {} # Number of users per site
    rtt = {}
    arrivalRateFromUserToSite = {}
    instances_per_app_site = {}
    instance_arrival_rate = {}
    lambdaLimits = {}

    # Define M as a large constant
    M = 1e6

    # Constraints and lambda for each app
    for app, app_data in applications.items():
        users = app_data["users"]
        request_rates = app_data["request_rates"]
        latency_site_user = {(i, j): app_data["latency_site_user"][f"('{i}', '{j}')"] for i in sites for j in users}
        service_rate = app_data["service_rate"]
        slo = app_data["slo"]

        x[app] = model.addVars(sites, users, vtype=GRB.BINARY, name=f"x_{app}")  # Assignment of users to sites
        y[app] = model.addVars(sites, vtype=GRB.BINARY, name=f"y_{app}")  # Site is open
        u[app] = model.addVars(sites, vtype=GRB.INTEGER, name=f"u_{app}")  # Number of users per site

        arrivalRateFromUserToSite[app] = model.addVars(sites, users, vtype=GRB.CONTINUOUS, name=f"arrivalRateFromUserToSite_{app}")  # arrival rate from user to site
        rtt[app] = model.addVars(sites, users, lb=-M, vtype=GRB.CONTINUOUS, name=f"rtt_{app}")  # Round trip time

        # Arrival rate at instance
        instance_arrival_rate[app] = model.addVars(sites, vtype=GRB.CONTINUOUS, name=f"utilization_{app}")
        # instance_per_app
        instances_per_app_site[app] = model.addVars(sites, vtype=GRB.INTEGER, name=f"instances_{app}")

        # Constraints

        # Each user is assigned to exactly one site
        model.addConstrs((x[app].sum('*', j) == 1 for j in users), name=f"assignToOne_{app}")

        # Users can only be linked to the open sites
        model.addConstrs((x[app][i, j] <= y[app][i] for i in sites for j in users), name=f"linkToOpen_{app}")

        # Enforce instance arrival rate
        model.addConstrs((arrivalRateFromUserToSite[app][i, j] == x[app][i, j] * request_rates[j] for i in sites for j in users), name=f"user_site_arrival_rate_{app}")

        # Arrival rate at the instances
        model.addConstrs((instance_arrival_rate[app][i] * instances_per_app_site[app][i] == arrivalRateFromUserToSite[app].sum(i, '*') for i in sites), name=f"instance_arrival_rate_{app}")
        
        # Number of users per site
        model.addConstrs((u[app][i] == x[app].sum(i, '*') for i in sites), name=f"numUsers_{app}")

        # If a site is open, it should have at least one instance deployed, no matter which application
        model.addConstrs((instances_per_app_site[app][i] >= y[app][i] for i in sites), name=f"atLeastOneInstanceIfOpen_{app}")

        # Lambda limits for each user and site
        lambdaLimits[app] = {i: {j: spo.brentq(lambda lm: waitingTimeDistr(slo - latency_site_user[i, j] - 1 / service_rate, lm, service_rate) - theta, 0.0, service_rate) for j in users} for i in sites}

        # Enforce maximum lambda
        for j in users:
            for i in sites:
                model.addConstr(x[app][i, j] * instance_arrival_rate[app][i] <= lambdaLimits[app][i][j], name=f"lambdaMax_{app}_{i}_{j}")

    #########################################
    # Global constraints for all the apps
    #########################################

    # Capacity constraints: number of instances of all application on one site shouldn't be bigger than the number of slots
    model.addConstrs((sum(instances_per_app_site[app][i] * applications[app]["slots"] for app in applications) <= capacities[i] for i in sites), name="capacity")

    # Link y_site with y[app][i], new!!
    for i in sites:
        for app in applications:
            model.addConstr(y[app][i] <= y_site[i], name=f"link_{app}_{i}")

    # Objective function
    if opti_pref == 1:
        # Minimize the number of open sites for all the applications
        # model.setObjective(sum(y[app][i] for app in applications for i in sites), GRB.MINIMIZE)
        model.setObjective(sum(y_site[i] for i in sites), GRB.MINIMIZE)
    elif opti_pref == 2:
        # model.setObjective(sum(instances_per_app_site[app].sum() for app in applications), GRB.MINIMIZE)
        model.setObjective(sum(instances_per_app_site[app][i] * applications[app]["slots"] for app in applications for i in sites), GRB.MINIMIZE)
    elif opti_pref == 3:
        # Minimize the financial cost
        model.setObjective(sum(instances_per_app_site[app][i] * applications[app]["slots"] * pricing[i] for app in applications for i in sites), GRB.MINIMIZE)
    else:
        print("No optimization preference selected.")
        sys.exit(1)

    return model, x, y, u, instance_arrival_rate, instances_per_app_site


# Optimize the model
def run_optimization(model):
    model.Params.OutputFlag = 0  # Suppress Gurobi output
    model.optimize()



def find_best_app_permutation(sites, capacities, pricing, opti_pref, applications, theta, pi=None):
    
    # Set up the initial minimum objective value based on optimization preference
    if opti_pref == 1:
        min_objval_permutations = len(sites)
    elif opti_pref == 2:
        min_objval_permutations = sum(capacities.values())
    else: 
        min_objval_permutations = sum(capacities[i] * pricing[i] for i in sites)

    apps_permutations = list(itertools.permutations(applications.keys()))

    if pi and len(apps_permutations) > pi:
        apps_permutations = random.sample(apps_permutations, pi)

    ordering_apps_final = None

    start_time = time.time()

    for apps_permutation in apps_permutations:
        remaining_capacities = capacities.copy()
        objval_each_permutation = 0

        for app in apps_permutation:
            single_app = {app: applications[app]}
            model, _, _, _, _, instances_per_app_site = initialize_model(
                sites, pricing, opti_pref, single_app, remaining_capacities, theta
            )
            run_optimization(model)

            if model.status == GRB.OPTIMAL:
                app_objval = model.objVal
                # print(f"Permutation {apps_permutation}, App {app} min cost: {app_objval}")
                objval_each_permutation += app_objval
                for i in sites:
                    remaining_capacities[i] -= instances_per_app_site[app][i].X * applications[app]["slots"]
            else:
                objval_each_permutation = float("inf")
                break
        # Have to be <= or it would miss the optimal solution
        # print(f"Total objective value for permutation {apps_permutation}: {objval_each_permutation}")
        if objval_each_permutation <= min_objval_permutations:
            min_objval_permutations = objval_each_permutation
            ordering_apps_final = apps_permutation

    elapsed_time = time.time() - start_time

    return ordering_apps_final, elapsed_time


def degrading_models(applications, sites, capacities, pricing, opti_pref, theta, exp_round_index, pi):

    degradation_state = {app: 0 for app in applications.keys()}
    fully_degraded_apps = set()

    active_model_name = {app: app for app in applications.keys()}

    priority_groups = {}
    for app, app_data in applications.items():
        priority_groups.setdefault(app_data["priority_group"], []).append(app)

    sorted_priorities = sorted(priority_groups.keys(), reverse=True)  # lowest priority first
    # Banning the highest priority from degradation, uncomment if needed
    # sorted_priorities = sorted_priorities[1:]  

    print(f"\n    ##### Degradation order based on priorities: {sorted_priorities} #####")

    for stage in range(1, len(sorted_priorities) + 1):

        active_priorities = sorted_priorities[:stage]
        affected_apps = [app for p in active_priorities for app in priority_groups[p]]
        print("\n===========================================================")
        print(f" ---- Stage {stage}: Degrading apps in priorities {active_priorities} ----")
        print(f"        Degrading applications: {affected_apps}")
        print("===========================================================\n")

        app_steps = {}         # Determine max degradation depth needed
        max_steps = 0

        for app in affected_apps:
            if "degrading_models" in applications[app]:
                levels = sorted([v["level"] for v in applications[app]["degrading_models"].values()])
                app_steps[app] = max(levels)
                max_steps = max(max_steps, app_steps[app])
                print(f"  • {app} can degrade to Level {app_steps[app]}")
            else:
                app_steps[app] = 0
                print(f"  • {app} has NO degrading models to degrade to, it remains at original level.")

        # Loop for the degradation levels, max_steps determined by the app with the deepest degradation path
        for step in range(1, max_steps + 1):

            print(f"\n====> Applying degradation LEVEL {step} for degradable apps:\n")

            for app in affected_apps:

                degrade_limit = app_steps[app]

                if degradation_state[app] == degrade_limit:
                    continue  

                if step <= degrade_limit and "degrading_models" in applications[app]:

                    for model_name, m in applications[app]["degrading_models"].items():

                        if m["level"] == step:

                            print(f"\033[1;33m |-> {app} is now degraded to: **{model_name}** (Level {step})\033[0m\n")

                            # Apply degraded configuration
                            applications[app]["slots"] = m["slots"]
                            applications[app]["slo"] = m["slo"]
                            applications[app]["service_rate"] = m["service_rate"]

                            active_model_name[app] = model_name

                            degradation_state[app] = step

                            # Mark if fully degraded now
                            if step == degrade_limit:
                                fully_degraded_apps.add(app)

            ordering_apps_final, _ = find_best_app_permutation(
                sites, capacities, pricing, opti_pref, applications, theta, pi
            )

            if ordering_apps_final is not None:

                print(f'\n\033[1;32m *** FEASIBLE SOLUTION FOUND after degradation! *** \033[0m\n')

                degraded_apps_list = [
                    active_model_name[app] for app, lvl in degradation_state.items() if lvl > 0
                ]

                affected_priorities = sorted(set(
                    applications[app]["priority_group"] for app in applications
                ))

                print(f" Degraded Models Used: {degraded_apps_list}")
                print(f" Active Model Names: {active_model_name}\n")
                
                return ordering_apps_final, degradation_state, degraded_apps_list, active_model_name, affected_priorities, fully_degraded_apps, priority_groups
            
            else:
                print(f'\n\033[1;31m No solution at this degradation level — increasing degradation... \033[0m\n')
                continue

    print("\n  NO FEASIBLE SOLUTION FOUND after full degradation.\n")
    return None, None, None, degradation_state, fully_degraded_apps, active_model_name

def print_results(
    sites, applications, x, y, u, instance_arrival_rate, instances_per_app_site, 
    theta, exp_round_index, degradation_introduced_apps=None, priority_levels=None, 
    active_model_name=None
):
    print("====================================================================================================")
    if degradation_introduced_apps and priority_levels is not None:
        print(f"**** Model degradation introduced for priority levels {', '.join(map(str, priority_levels))}: {', '.join(degradation_introduced_apps)} ****")
        print(f"Model degradation details:")

    for i in sites:
        if all(y[app][i].X < 0.5 for app in applications):
            print("----------------------------------------------------------------------------------------------------")
            print(f"Site {i} is closed.")

        for app, app_data in applications.items():
            if y[app][i].X > 0.5:
                if active_model_name is not None:
                    display_name = active_model_name.get(app, app)
                else:
                    display_name = app
                service_rate = app_data["service_rate"]
                slo = app_data["slo"]
                users = app_data["users"]
                latency_site_user = {(i, j): app_data["latency_site_user"][f"('{i}', '{j}')"] for j in users}
                rho = instance_arrival_rate[app][i].X / service_rate

                print("----------------------------------------------------------------------------------------------------")
                print(f"Site {i} is open for app: {display_name} with {math.ceil(instances_per_app_site[app][i].X)} instances, {u[app][i].X} users assigned.")
                print(f"    Arrival rate at the instances for *{display_name}*: {instance_arrival_rate[app][i].X:.5f}, Utilization of instances: {rho:.5f}")
                print(f"*** SLO used for {display_name}: {applications[app]['slo']} ***")

                for j in users:
                    if x[app][i, j].X > 0.5:                    
                        rtt_value = latency_site_user[i, j] + 1 / service_rate + rho / (2 * service_rate * (1 - rho))
                        perc = spo.brentq(lambda t: waitingTimeDistr(t - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate) - theta, 0, slo + 1)
                        print(f"  User {j} with average RTT {rtt_value} and {theta * 100}% percentile RTT {perc}.")
                        print(f"    Probability that the requests of this user will stay below the SLT-RTT: {waitingTimeDistr(slo - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate)}")

    print("====================================================================================================")

    # # Generate placement plan JSON
    # placement_plan_json(sites, applications, x, y, instances_per_app_site, exp_round_index, active_model_name=active_model_name)


def update_universal_placement_plan(universal_placement_plan, sites, applications, x, y, instances_per_app_site):
    """Update the universal placement plan with optimization results"""
    for i in sites:
        for app in applications:
            if y[app][i].X > 0.5:
                app_info = {
                    "app": app,  # Store original app key
                    "instances": math.ceil(instances_per_app_site[app][i].X),
                    "users": []
                }
                for j in applications[app]["users"]:
                    if x[app][i, j].X > 0.5:
                        user_info = {
                            "user": j,
                            "request_rate": applications[app]["request_rates"][j]
                        }
                        app_info["users"].append(user_info)
                universal_placement_plan["sites"][i]["applications"].append(app_info)

def placement_plan_json(universal_placement_plan, exp_round_index, active_model_name=None):
    """Generate the final placement plan JSON file"""
    final_placement_plan = {"sites": []}
    
    for site, site_data in universal_placement_plan["sites"].items():
        site_info = {
            "site": site,
            "applications": []
        }
        
        # Apply active_model_name mapping to each application
        for app_entry in site_data["applications"]:
            app_key = app_entry["app"]
            display_name = active_model_name.get(app_key, app_key) if active_model_name else app_key
            
            app_info = {
                "app": display_name,  # Use degraded model name if available
                "instances": app_entry["instances"],
                "users": app_entry["users"]
            }
            site_info["applications"].append(app_info)
        
        final_placement_plan["sites"].append(site_info)
    
    # Write the final placement plan to a JSON file
    with open(f"placement_plan_r{exp_round_index}.json", "w") as json_file:
        json.dump(final_placement_plan, json_file, indent=4)
    
    print(f"Placement plan has been saved to placement_plan_r{exp_round_index}.json")


def main ():

    parser = argparse.ArgumentParser(description='Run the OCCAMY optimization.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--round', type=int, required=True, help='The round index for the output file.')
    args = parser.parse_args()

    exp_round_index = args.round
    json_file = args.input

    with open(json_file, 'r') as f:
        data = json.load(f)

    sites = data["sites"]
    capacities = data["capacities"]
    pricing = data["pricing"]
    opti_pref = data["opti_pref"]
    applications = data["applications"]
    theta = 0.99 

    # Sampling of the permutations
    num_apps = len(applications)
    pi = int(math.factorial(num_apps))

    starting_time = time.time()

    degradation = False
    universal_placement_plan = {
        "sites": {site: {"site": site, "applications": []} for site in sites}
    }


    # model, x, y, u, instance_arrival_rate, instances_per_app_site = initialize_model(sites, pricing, opti_pref, applications, capacities, theta)
    ordering_apps_final, time_permutation = find_best_app_permutation(
        sites, capacities, pricing, opti_pref, applications, theta, pi
    )
    

    active_model_name = {app: app for app in applications.keys()}  # Initialize for both paths
    degraded_apps_list = []
    affected_priorities = []
    fully_degraded_apps = set()
    priority_groups = {}

    if ordering_apps_final is not None:
        print(f'\n\033[1;32m *** FEASIBLE SOLUTION FOUND without degradation! *** \033[0m\n')
        # print_results(sites, applications, x, y, u, instance_arrival_rate, instances_per_app_site, theta, exp_round_index)
    else:
        print("\n**** No feasible solution found in the original model. Degradation in process...****")
        ordering_apps_final, degradation_state, degraded_apps_list, active_model_name, affected_priorities, fully_degraded_apps, priority_groups = degrading_models(applications, sites, capacities, pricing, opti_pref, theta, exp_round_index, pi)
        if ordering_apps_final is not None:
            degradation = True
        else:
            print("\n**** No feasible solution found even after full degradation. ****")
            sys.exit(1)
    ending_time = time.time()

    
    objval_min = 0
    remaining_capacities = capacities.copy()

    for app in ordering_apps_final:
        single_app = {app: applications[app]}
        model, x, y, u, instance_arrival_rate, instances_per_app_site = initialize_model(
            sites, pricing, opti_pref, single_app, remaining_capacities, theta
        )
        run_optimization(model)

        if model.status == GRB.OPTIMAL:
            if degradation:
                print_results(sites, single_app, x, y, u, instance_arrival_rate, instances_per_app_site, theta, exp_round_index, degraded_apps_list, affected_priorities, active_model_name)
            else :
                print_results(sites, single_app, x, y, u, instance_arrival_rate, instances_per_app_site, theta, exp_round_index)
            objval_min += model.objVal

            for i in sites:
                remaining_capacities[i] -= sum(instances_per_app_site[app][i].X * applications[app]["slots"] for app in single_app)
            update_universal_placement_plan(universal_placement_plan, sites, single_app, x, y, instances_per_app_site)


    # At the end of main(), after all the results are printed:
    print("====================================================================================================")
    print(f"Evaluated {pi} permutations with initial SLOs in {time_permutation:.2f} seconds")
    if not degradation:
        print("No degradation needed.")
        placement_plan_json(universal_placement_plan, exp_round_index)  # No active_model_name
    else:
        print("Degradation applied:")
        print(f"Degraded priority levels {', '.join(map(str, affected_priorities))} with applications:")
        for app in applications.keys():  # Iterate over original app keys
            if degradation_state[app] > 0:  # Check if this app was degraded
                degraded_model_name = active_model_name[app]  # Get the degraded model name
                if app in fully_degraded_apps:
                    print(f"  - '{app}' FULLY degraded to '{degraded_model_name}' (level {degradation_state[app]}).")
                else:
                    print(f"  - '{app}' degraded to '{degraded_model_name}' (level {degradation_state[app]}).")
        placement_plan_json(universal_placement_plan, exp_round_index, active_model_name)

    actual_model_names = [active_model_name[app] for app in ordering_apps_final]
    print(f"Permutation {actual_model_names} with minimum objective value: {objval_min}")
    print(f"Total time cost: {ending_time - starting_time:.2f} seconds")
    print("====================================================================================================")

if __name__ == "__main__":
    main()