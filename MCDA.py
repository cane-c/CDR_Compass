import pandas as pd
import excel as exl
import evaluations as ev
from other.dictToArray import transform_to_Array
import visualisations as viz
import topsis as tp
import other as other
import compromise_programming as cp
import numpy as np



###########################################################################
######################### DIRECT RATING ###################################
###########################################################################

# Reading exel data - performance of evaluations:
expert1_eval = exl.ExpertEvaluation('Test1.xlsx')
evaluation_data_e1 = expert1_eval.get_evaluation()
expert2_eval = exl.ExpertEvaluation('Test2.xlsx')
evaluation_data_e2 = expert2_eval.get_evaluation()
#print(evaluation_data_e1)

### Aggregation of expert evaluations:
expertMatrix = ev.aggregateEvaluations(evaluation_data_e1, evaluation_data_e2)
#print(expertMatrix)

### Show aggregated expert Matrix
#viz.showExpertMatrix(expertMatrix)

#################################################################
## SCENARIO 1: Compare Actions. Internal Ref values #############
#################################################################

### Calculate Deterministic TOPSIS scenarios with PIS and NIS between data
#topsisDeterministicData = tp.topsis_deterministic(expertMatrix, False)
#print(topsisDeterministicData)

# Sorting in descending order
#ranked_data = dict(sorted(topsisDeterministicData["max"].items(), key=lambda item: item[1], reverse=True))
#print(ranked_data)
#print(dict(sorted(topsisDeterministicData["mean"].items(), key=lambda item: item[1], reverse=True)))


### Show deterministic TOPSIS results
#viz.showDeterministicTopsis(topsisDeterministicData)


### Calculate Stochastic TOPSIS scenarios with PIS and NIS between data and likelihood of ranking
#topsisStochasticData = tp.topsis_MC(expertMatrix, 1000, False)
#print(topsisStochasticData)

### Show stochastic TOPSIS results
#viz.draw_table(topsisStochasticData)

#################################################################
## SCENARIO 2: Compare Actions. External Ref values #############
#################################################################

### Provide PIS and NIS
ref_values = {'PIS': 10, 'NIS' : 0}

#topsisDeterministicData2 = tp.topsis_deterministic(expertMatrix, False, ref_values)
#print(topsisDeterministicData2)

# Sorting in descending order
#ranked_data = dict(sorted(topsisDeterministicData2["mean"].items(), key=lambda item: item[1], reverse=True))
#print(ranked_data)

### Show deterministic TOPSIS results
#viz.showDeterministicTopsis(topsisDeterministicData2)

### Calculate Stochastic TOPSIS scenarios with PIS and NIS between data
topsisStochasticActions = tp.topsis_MC_Actions(expertMatrix, n=1000, normalize=False, ref_values=ref_values, criteria_to_analyze=None, distribution="negative_exponential" )
viz.showScatterMC_dictionary(topsisStochasticActions)

### Calculate Stochastic TOPSIS scenarios with PIS and NIS between data and likelihood of ranking
#topsisStochasticData2 = tp.topsis_MC(expertMatrix, 2000, False, ref_values=ref_values, criteria_to_analyze=None, distribution="negative_exponential")
#print(topsisStochasticData2)

### Show stochastic TOPSIS results
#viz.draw_table(topsisStochasticData2)

#################################################################
## SCENARIO 3: Compare Scenarios by selecting actions ###########
#################################################################

### define which actions should belong to a strategy
strategy1 = {'A1', 'A2', 'A3'}
strategy2 = {'A3', 'A4'}
strategy3 = {'A1', 'A2', 'A3', 'A5', 'A6', 'A7'}

# Define strategies
strategies = {
    'strategy1': {'A1', 'A2', 'A3'},
    'strategy2': {'A3', 'A4'},
    'strategy3': {'A1', 'A2', 'A3', 'A5'}
}

#internal aggregation; Randomize raw evaluation scores for each action. Then, average criterion-specific evaluations before deriving closeness indices for each strategy
"""
result_interanl_stochastic = tp.topsis_MC_Strategies_From_Actions(
    dataset=expertMatrix,
    strategies=strategies,
    n=1000,
    normalize=False,
    distribution="positive_exponential"
)
"""
### Internal aggregation of the performance evaluations of actions into min, max and mean (aggregation of alternative performanace scores)
"""
aggregateActionEval = uses arithmetic mean of min, max and mean method
aggregateActionEval_geometricMean = uses geometric mean for min, max and mean values
aggregateActionEval_sum = uses sum for min, max and mean values
"""
"""
strategy1Matrix = ev.aggregateActionEval(expertMatrix, strategy1)
strategy2Matrix = ev.aggregateActionEval(expertMatrix, strategy2)
strategy3Matrix = ev.aggregateActionEval(expertMatrix, strategy3)
"""

### External aggregation of the performance evaluations of actions into min, max and mean (aggregation of alternative closeness index scores)
#strategy1Matrix = ev.aggregateActionsExt_Average(topsisDeterministicData2, strategy1)
#strategy2Matrix = ev.aggregateActionsExt_Average(topsisDeterministicData2, strategy2)
#strategy3Matrix = ev.aggregateActionsExt_Average(topsisDeterministicData2, strategy3)

### Combine strategies into a dictionary
# combine_strategies -  for internal aggregation method
#combine_Ext_Strategies - for external aggregation method
#strategiesDictionary = other.combine_strategies(strategy1Matrix, strategy2Matrix, strategy3Matrix)
#print("STRATEGIES MATRIX:")
#print(strategiesDictionary)

### Calculate deterministic closeness results of strategies of internal aggregation
#topsisDeterministicStrategies = tp.topsis_deterministic(strategiesDictionary, False, ref_values=ref_values)
#print(topsisDeterministicStrategies)

# Visualise either topsisDeterministicStrategies or strategiesDictionary
#viz.showDeterministicTopsis(topsisDeterministicStrategies)


### Perform MC for one strategy and visualise it in a scatter plot to show the spread of closeness to ideal.
### This method needs to have the ref values defined
#topsisMC_strategy1 = tp.topsis_MC_Strategy(expertMatrix, strategy=strategy1, n=1000, normalize=False)
#print(topsisMC_strategy)
#topsisMC_strategy2 = tp.topsis_MC_Strategy(expertMatrix, strategy=strategy2, n=1000, normalize=False)
#topsisMC_strategy3 = tp.topsis_MC_Strategy(expertMatrix, strategy=strategy3, n=1000, normalize=False)
#viz.showScatterMC(topsisMC_strategy1, topsisMC_strategy2, topsisMC_strategy3)

### Perform MC for strategies (dictionary as an input. Refernce values could be provided or not)
# Internal distribution: Using a matrix for each strategy where each criterion is associated with its average minimum, maximum, and mean values across actions, randomise evaluation score for each criterion. 
# Then, calculate closeness indices for each strategy. 
# distribution : By default it is a uniform. Other values: "positive_exponential", "negative_exponential", "normal"


#topsisMC_strategies = tp.topsis_MC_Strategies(strategiesDictionary, 1000, False, ref_values=ref_values, criteria_to_analyze=None, distribution="uniform")
#print (topsisMC_strategies)

#viz.showScatterMC_dictionary(result_interanl_stochastic)

###############################################################################################################
## SCENARIO 4: Compare Scenarios according to criteria families - scattered diagrams when 2 dimensions  #######
###############################################################################################################

### define criteria families. They MUST have 'total' included and use a LIST
criteria_families = {
    "env": ["C1", "C2"],
    "soc": ["C3", "C4"],
    "total": ["C1", "C2", "C3", "C4"]
}

### generate stochastic outputs for each criteria families
# strategiesDict has to be from internal aggregation!
#topsisMC_strat_families = tp.topsis_MC_criteriaFamilies_Strat(strategiesDictionary, 1000, False, ref_values = ref_values, criteria_families=criteria_families)
#print(topsisMC_strat_families)
#viz.showCompareFamilies(topsisMC_strat_families, 'c_i_env', 'c_i_soc') #with scattered data

##########################################################################################
## SCENARIO 5: Calculate TOPSIS by doing internal aggregation of expert views ############
##########################################################################################
 
#topsisDeterm_Internal = tp.topsis_determ_internal_aggregation(evaluation_data_e1, evaluation_data_e2, normalize=False, ref_values=ref_values)
#print (topsisDeterm_Internal)
#viz.show_average_ci_table(topsisDeterm_Internal)

##########################################################################################
## SCENARIO 6: Find best performing alternatives on C with max closeness index ###########
##########################################################################################

#topsisDeterm_Internal_env = tp.topsis_determ_internal_aggregation(evaluation_data_e1, evaluation_data_e2, normalize=False, ref_values=ref_values, criteria_to_analyze=("C1", "C2"))
#topsisDeterm_Internal_soc = tp.topsis_determ_internal_aggregation(evaluation_data_e1, evaluation_data_e2, normalize=False, ref_values=ref_values, criteria_to_analyze=("C3", "C4"))

#viz.plot_best_alternatives(expertMatrix, topsisDeterm_Internal["average_ci"])
#viz.scatter_best_alternatives(expertMatrix, topsisDeterm_Internal_env["average_ci"], topsisDeterm_Internal_soc["average_ci"] )

##########################################################################################
## SCENARIO 7:Compromise programming (externally aggregated values)            ###########
##########################################################################################
#p = float('inf') # Chebyshev distance where ther is no compensability
#p = 1 #manhattan distance - full compensability
#p=2 #euclidean distance - partial compensabiity
#compromiseProg = cp.cp_deterministic(expertMatrix, ref_values, form = 3, p = p)
#print(compromiseProg)
#ranked_data = dict(sorted(compromiseProg["min"].items(), key=lambda item: item[1], reverse=False))
#print(ranked_data)
#ranked_data = dict(sorted(compromiseProg["max"].items(), key=lambda item: item[1], reverse=False))
#print(ranked_data)
#ranked_data = dict(sorted(compromiseProg["mean"].items(), key=lambda item: item[1], reverse=False))
#print(ranked_data)
#viz.showDeterministicTopsis(compromiseProg)

##########################################################################################
## SCENARIO 8: Calculate CP by doing internal aggregation of expert views ############
##########################################################################################
 
#cp_determ_Internal = cp.cp_determ_internal_aggregation(evaluation_data_e1, evaluation_data_e2, ref_values=ref_values, form = 1, p = p)
"""
print (cp_determ_Internal)
print("mean")
print(dict(sorted(cp_determ_Internal["average"].items(), key=lambda item: item[1], reverse=False)))
print("min")
print(dict(sorted(cp_determ_Internal["min"].items(), key=lambda item: item[1], reverse=False)))
print("max")
print(dict(sorted(cp_determ_Internal["max"].items(), key=lambda item: item[1], reverse=False)))
"""

##########################################################################################
## SCENARIO 9: Compare actions on different criterion families ############
##########################################################################################
#topsisStochasticActions_env = tp.topsis_MC_Actions(expertMatrix, n=1000, normalize=False, ref_values=ref_values, criteria_to_analyze=("C1", "C2"), distribution="normal" )
#topsisStochasticActions_soc = tp.topsis_MC_Actions(expertMatrix, n=1000, normalize=False, ref_values=ref_values, criteria_to_analyze=("C3", "C4"), distribution="normal" )
#viz.showCompareActions(topsisStochasticActions_env, topsisStochasticActions_soc)

##########################################################################################
## SCENARIO 10: Calculate CP for strategies ############
##########################################################################################
"""
cp_strategies = cp.cp_deterministic(strategiesDictionary, ref_values=ref_values, form = 1, p = p)
print(cp_strategies)
print("mean")
print(dict(sorted(cp_strategies["mean"].items(), key=lambda item: item[1], reverse=False)))
print("min")
print(dict(sorted(cp_strategies["min"].items(), key=lambda item: item[1], reverse=False)))
print("max")
print(dict(sorted(cp_strategies["max"].items(), key=lambda item: item[1], reverse=False)))
"""


###########################################################################
######################### INTERVAL RATING ###################################
###########################################################################

# Reading exel data - performance of evaluations:
"""
expert1_eval = exl.Interval_ExpertEvaluation('ev1.xlsx')
evaluation_data_e1 = expert1_eval.get_evaluation()
expert2_eval = exl.Interval_ExpertEvaluation('ev2.xlsx')
evaluation_data_e2 = expert2_eval.get_evaluation()
expert3_eval = exl.Interval_ExpertEvaluation('ev3.xlsx')
evaluation_data_e3 = expert3_eval.get_evaluation()
expert4_eval = exl.Interval_ExpertEvaluation('ev4.xlsx')
evaluation_data_e4 = expert4_eval.get_evaluation()
"""
#expert5_eval = exl.Interval_ExpertEvaluation('ev5.xlsx')
#evaluation_data_e5 = expert5_eval.get_evaluation()

#print(evaluation_data_e1)

######################################################
############## DETERMINISTIC AGGREGATION
#######################################################

#expertMatrix_Int = ev.aggregateEvaluations_Int(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4)
#print(expertMatrix_Int)

#print results to Excel
#other.matrixExel(expertMatrix_Int=expertMatrix_Int)

#viz.showExpertMatrix_Int(expertMatrix_Int, selected_criteria=["EN1", "EN2", "EN3"])

#######################################################
################STOCHASTIC AGGREGATION
#######################################################

#expertMatrix_Int_Stochastic = ev.aggregate_MC_Int(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, n = 1000)

#visualisise results for a selected Action
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A4")
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A6")
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A9")
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A10")
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A11")
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A12")
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A16")
#viz.visualize_MC_Int(expertMatrix_Int_Stochastic, "A21")



#################################################################
## SCENARIO 1: Compare Actions. External Ref values. Deterministic TOPSIS modelling #############
#################################################################




#topsisDeterministicData_Int = tp.topsis_deterministic_Int(expertMatrix_Int, ref_values=ref_values,  normalize=False, criteria_to_analyze=econ)

#viz.showDeterministicTopsis(topsisDeterministicData_Int)


#################################################################
## SCENARIO 2: Compare Actions. External Ref values. Stochastic TOPSIS modelling #############
#################################################################

#topsisMC_Data_Int = tp.topsis_MC_Int_Actions(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, n=2000, criteria_to_analyze=econ)

#viz.showScatterMC_dictionary(topsisMC_Data_Int)

#################################################################
## SCENARIO 3: Probabilities of rankings. External Ref values. Stochastic TOPSIS modelling #############
#################################################################

#rankings_Int = tp.topsis_MC_Int_Ranking(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, n=1000, criteria_to_analyze=econ)
#print(rankings_Int)

#viz.draw_table(rankings_Int)

#################################################################
## SCENARIO 4: Trade-offs of actions #############
#################################################################

#topsisMC_Data_Int_env = tp.topsis_MC_Int_Actions(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, n=1000, criteria_to_analyze=env)
#topsisMC_Data_Int_soc = tp.topsis_MC_Int_Actions(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, n=1000, criteria_to_analyze=soc)
#topsisMC_Data_Int_econ = tp.topsis_MC_Int_Actions(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, n=1000, criteria_to_analyze=econ)

#rename in the function x and y axis, first is currently environmental and second social
#viz.showCompareActions(topsisMC_Data_Int_env, topsisMC_Data_Int_soc)
#viz.showCompareActions(topsisMC_Data_Int_env, topsisMC_Data_Int_econ)
#viz.showCompareActions(topsisMC_Data_Int_soc, topsisMC_Data_Int_econ)


#################################################################
## SCENARIO 5: Cumulative impacts #############
#################################################################

#strat_MC_all_soc = tp.topsis_MC_Int_Strategies(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, l=1000, actions_to_analyze=["A4", "A10", "A11", "A16"], criteria_to_analyze=soc)
#strat_MC_all_env = tp.topsis_MC_Int_Strategies(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, l=1000, actions_to_analyze=["A4", "A10", "A11", "A16"], criteria_to_analyze=env)
#strat_MC_all_econ = tp.topsis_MC_Int_Strategies(evaluation_data_e1, evaluation_data_e2, evaluation_data_e3, evaluation_data_e4, ref_values=ref_values, l=1000, actions_to_analyze=["A4", "A10", "A11", "A16"], criteria_to_analyze=econ)

#viz.plot_3d_scatter(strat_MC_all_env, strat_MC_all_soc, strat_MC_all_econ)

