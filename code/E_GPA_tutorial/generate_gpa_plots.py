# generate_gpa_plots.py
import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score

def perf_eval_fn(y_pred,y,**kwargs):
	performance_metric = kwargs['performance_metric']
	if performance_metric == 'log_loss':
		return log_loss(y,y_pred)
	elif performance_metric == 'accuracy':
		return accuracy_score(y,y_pred > 0.5)

def main():
	# Parameter setup
	run_experiments = True
	make_plots = False
	save_plot = False
	include_legend = False
	constraint_name = 'disparate_impact'
	fairlearn_constraint_name = constraint_name
	fairlearn_epsilon_eval = 0.8 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_eval_method = 'two-groups' # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_epsilons_constraint = [0.01,0.1,1.0] # the epsilons used in the fitting constraint
	performance_metric = 'accuracy'
	n_trials = 50
	data_fracs = np.logspace(-4,0,15)
	n_workers = 1
	results_dir = f'results/gpa_{constraint_name}_{performance_metric}_2022Sep28_debug'
	plot_savename = os.path.join(results_dir,f'gpa_{constraint_name}_{performance_metric}.png')

	verbose=True

	# Load spec
	specfile = f'../interface_outputs/gpa_{constraint_name}/spec.pkl'
	spec = load_pickle(specfile)
	spec.optimization_hyperparams['num_iters'] = 100

	os.makedirs(results_dir,exist_ok=True)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	label_column = dataset.label_column
	include_sensitive_columns = dataset.include_sensitive_columns

	test_features = dataset.df.loc[:,
		dataset.df.columns != label_column]
	test_labels = dataset.df[label_column]

	if not include_sensitive_columns:
		test_features = test_features.drop(
			columns=dataset.sensitive_column_names) 

	# Setup performance evaluation function and kwargs 
	# of the performance evaluation function

	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		'performance_metric':performance_metric
		}

	plot_generator = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		n_workers=n_workers,
		datagen_method='resample',
		perf_eval_fn=perf_eval_fn,
		constraint_eval_fns=[],
		results_dir=results_dir,
		perf_eval_kwargs=perf_eval_kwargs,
		)

	# # Baseline models
	if run_experiments:
		plot_generator.run_baseline_experiment(
			model_name='random_classifier',verbose=True)

		plot_generator.run_baseline_experiment(
			model_name='logistic_regression',verbose=True)

		# Seldonian experiment
		plot_generator.run_seldonian_experiment(verbose=verbose)


	######################
	# Fairlearn experiment 
	######################

	# fairlearn_sensitive_feature_names=['M']
	
	# # Make dict of test set features, labels and sensitive feature vectors
	
	# # Make dict of test set features, labels and sensitive feature vectors
	if 'offset' in test_features.columns:
		test_features_fairlearn = test_features.drop(columns=['offset'])
	else:
		test_features_fairlearn = test_features
	fairlearn_eval_kwargs = {

		'X':test_features_fairlearn,
		'y':test_labels,
		'sensitive_features':dataset.df.loc[:,
			fairlearn_sensitive_feature_names],
		'eval_method':fairlearn_eval_method,
		'performance_metric':performance_metric,
		}

	if run_experiments:
		for fairlearn_epsilon_constraint in fairlearn_epsilons_constraint:
			plot_generator.run_fairlearn_experiment(
				verbose=verbose,
				fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
				fairlearn_constraint_name=fairlearn_constraint_name,
				fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
				fairlearn_epsilon_eval=fairlearn_epsilon_eval,
				fairlearn_eval_kwargs=fairlearn_eval_kwargs,
				)

	if make_plots:
		if save_plot:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric,
				include_legend=include_legend,
				savename=plot_savename)
		else:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				include_legend=include_legend,
				performance_label=performance_metric)



if __name__ == "__main__":
	main()