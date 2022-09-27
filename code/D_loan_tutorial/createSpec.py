# createSpec.py
import os
from seldonian.parse_tree.parse_tree import (ParseTree,
    make_parse_trees_from_constraints)

from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json,save_pickle,
    load_supervised_metadata)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import LogisticRegressionModel
from seldonian.models import objectives

if __name__ == '__main__':
    data_pth = "../../static/datasets/supervised/german_credit/german_loan_numeric_forseldonian.csv"
    metadata_pth = "../../static/datasets/supervised/german_credit/metadata_german_loan.json"
    save_dir = '../../../interface_outputs/loan_disparate_impact_seldodef'
    os.makedirs(save_dir,exist_ok=True)
    # Load metadata
    metadata_dict = load_json(metadata_pth)

    (regime, sub_regime, columns,
        sensitive_columns) = load_supervised_metadata(metadata_pth)
    
    # Use logistic regression model
    model = LogisticRegressionModel()
    
    # Set the primary objective to be log loss
    primary_objective = objectives.logistic_loss

    # Load dataset from file
    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        include_sensitive_columns=False,
        include_intercept_term=False,
        file_type='csv')
    
    # Define behavioral constraints
    constraint_strs = ['min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M])) >= 0.9'] 
    
    # For each constraint (in this case only one), make a parse tree
    deltas = [0.05]
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime='supervised_learning',
        sub_regime='classification',columns=columns)

    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='classification',
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        initial_solution_fn=model.fit,
        use_builtin_primary_gradient_fn=True,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : 0.5,
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 1500,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    spec_save_name = os.path.join(save_dir,'spec.pkl')
    save_pickle(spec_save_name,spec)
    print(f"Saved Spec object to: {spec_save_name}")