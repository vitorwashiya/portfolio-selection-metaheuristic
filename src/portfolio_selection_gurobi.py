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
        self.w = [
            self.model.addVar(name=str(i), vtype=GRB.CONTINUOUS, lb=0.0)
            for i in range(self.num_ast)
        ]

    def add_objective_function(self) -> None:
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
        self.model.addConstr(sum(self.w) == 1, name="full investment")

    def optimize(self):
        self.add_weights()
        self.add_objective_function()
        self.add_constraints()
        self.model.optimize()
        return np.array([v.x for v in self.model.getVars()])
