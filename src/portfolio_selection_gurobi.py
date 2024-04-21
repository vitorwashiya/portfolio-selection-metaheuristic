from gurobipy import Model, GRB
import pandas as pd
import numpy as np

RISK_AVER = 0.5


class PortfolioSelectionGurobi:
    '''
    Portfolio Selection using Gurobi - Minimization problem

    Parameters:
    - data: Dataframe with daily stock returns (index="Date", columns=["Stock.Close"])
    - num_ast: Number of assets to be selected (Default = None)
    - exp_ret_df: Expected Returns Dataframe (Default = None)
    - cov_mat: Covariance Matrix Dataframe (Default = None)
    - risk_aver: Risk Aversion Coefficient (Default = 0.5)
    - norm_param: Normalization Parameters with the following keys: ret_min, ret_max, var_min, var_max (Default = None)
    '''

    def __init__(self,
                 data: pd.DataFrame,
                 num_ast: int = None,
                 exp_ret_df: np.ndarray = None,
                 cov_mat: np.ndarray = None,
                 risk_aver: float = RISK_AVER,
                 norm_param: dict = None):

        self.num_ast = num_ast if num_ast else data.shape[1]
        self.exp_ret_df = exp_ret_df if exp_ret_df else data.mean().values
        self.cov_mat = cov_mat if cov_mat else data.cov().values
        self.risk_aver = risk_aver
        self.norm_param = norm_param
        self.model = Model(name="linear program")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("NonConvex", 2)
        self.w = None

    def add_weights(self) -> None:
        """
        Adds weight variables to the model.

        This method adds weight variables to the optimization model. Each weight variable represents the allocation
        of a specific asset in the portfolio. The weight variables are continuous and have a lower bound of 0.0.

        Returns:
            None
        """
        self.w = [
            self.model.addVar(name=str(i), vtype=GRB.CONTINUOUS, lb=0.0)
            for i in range(self.num_ast)
        ]

    def add_objective_function(self) -> None:
        """
        Adds the objective function to the optimization model.

        The objective function is a combination of the expected return and the risk of the portfolio.
        The function is minimized based on the risk aversion parameter.

        Returns:
            None
        """
        w_exp_ret = self.exp_ret_df @ self.w
        w_var = self.w @ self.cov_mat @ self.w

        if self.norm_param:
            w_exp_ret = (w_exp_ret - self.norm_param["ret_min"]) / (
                self.norm_param["ret_max"] - self.norm_param["ret_min"])
            w_var = (w_var - self.norm_param["var_min"]) / (
                self.norm_param["var_max"] - self.norm_param["var_min"])

        function = self.risk_aver * w_var + (self.risk_aver - 1) * w_exp_ret
        self.model.setObjective(function, GRB.MINIMIZE)

    def add_constraints(self) -> None:
        """
            Adds constraints to the optimization model.

            This method adds a constraint to ensure that the sum of all weights in the portfolio is equal to 1,
            which represents the requirement of full investment.

            Returns:
                None
            """
        self.model.addConstr(sum(self.w) == 1, name="full investment")

    def optimize(self) -> np.ndarray:
        """
            Optimize the portfolio selection problem.

            This method adds weights, objective function, constraints, and then optimizes the model.
            It returns an array of the optimized variable values.

            Returns:
                numpy.ndarray: Array of optimized variable values.
            """
        self.add_weights()
        self.add_objective_function()
        self.add_constraints()
        self.model.optimize()
        return np.array([v.x for v in self.model.getVars()])
