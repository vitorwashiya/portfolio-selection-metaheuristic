import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from typing import Union


class PerformanceTracker:
    '''
    A class that calculates various performance metrics for a portfolio.
    
    Parameters:
    - data (pd.Series): The returns of the portfolio.
    - mkt_ret (pd.Series, optional): The returns of the market index. Default is None.
    - risk_free (float, optional): The annual risk-free rate. Default is 0.
    - alph_var (float, optional): The significance level for value-at-risk calculation. Default is 0.05.
    - period (str, optional): The period of the returns data. Can be either "daily" or "weekly". Default is "daily".

    Example usage:
    ```
    data_returns = pd.Series([0.01, 0.02, -0.03, 0.04, 0.05])
    market_returns = pd.Series([0.02, 0.03, -0.01, 0.02, 0.01])
    tracker = PerformanceTracker(data_returns, mkt_ret=market_returns)
    result = tracker()
    print(result)
    ```
    '''

    def __init__(self,
                 data: pd.Series,
                 mkt_ret: Union[pd.Series, None] = None,
                 risk_free: float = 0,
                 alph_var: float = 0.05,
                 period: str = "daily"):
        self.data = data
        self.mkt_ret = mkt_ret
        self.risk_free = risk_free
        self.alph_var = alph_var
        self.period = period
        if period not in ["daily", "weekly"]:
            raise Exception("Period must be either daily or weekly.")

    def annualized_return(self, data: pd.Series):
        """
        Calculates the annualized return of a given dataset.

        Parameters:
        - data (pandas.Series or pandas.DataFrame): The dataset containing the returns.

        Returns:
        - float: The annualized return.
        """
        cuml_ret = (1 + data).prod()
        years = len(data) / 252 if self.period == "daily" else len(data) / 52
        annual_ret = (cuml_ret**(1 / years)) - 1
        return annual_ret

    def annualized_std_return(self):
        """
        Calculate the annualized standard deviation of returns.

        Returns:
        - float: The annualized standard deviation of returns, expressed as a percentage.
        """
        std_dev = self.data.std()
        if self.period == "daily":
            return std_dev * np.sqrt(252)
        else:
            return std_dev * np.sqrt(52)

    def portfolio_beta(self):
        """
        Calculates the beta of the portfolio.

        The beta measures the sensitivity of the portfolio's returns to the market returns.
        It is calculated as the covariance between the portfolio returns and the market returns,
        divided by the variance of the market returns.

        Returns:
          float: The beta of the portfolio.
          None: If the market returns are not provided.
        """
        if isinstance(self.mkt_ret, pd.Series):
            cov = np.cov(self.data, self.mkt_ret, ddof=0)[0, 1]
            mkt_var = np.var(self.mkt_ret, ddof=0)
            return cov / mkt_var
        else:
            return None

    def expected_return_capm(self):
        """
        Calculates the expected return of a portfolio using the Capital Asset Pricing Model (CAPM).

        The CAPM formula is given by:
          capm_returns = annual_risk_free + beta * (market_expected_returns - annual_risk_free)

        Returns:
          The expected return of the portfolio calculated using the CAPM formula.
        """
        mkt_exp_ret = self.annualized_return(data=self.mkt_ret)
        beta = self.portfolio_beta()
        return self.risk_free + beta * (mkt_exp_ret - self.risk_free)

    def calculate_portfolio_alpha(self):
        """
        Calculates the portfolio alpha.

        The portfolio alpha is calculated as the difference between the annualized return
        and the expected return based on the Capital Asset Pricing Model (CAPM).

        Returns:
          float: The portfolio alpha.
        """
        return max(
            0,
            self.annualized_return(data=self.data) -
            self.expected_return_capm())

    def sharpe_ratio(self):
        """
        Calculates the Sharpe ratio, which measures the risk-adjusted return of an investment strategy.

        The Sharpe ratio is calculated as the difference between the annualized return and the risk-free rate,
        divided by the annualized standard deviation of the return.

        Returns:
          The Sharpe ratio of the investment strategy.
        """
        return (self.annualized_return(data=self.data) -
                self.risk_free) / self.annualized_std_return()

    def max_drawdown(self):
        """
        Calculates the maximum drawdown of the portfolio.

        The maximum drawdown is defined as the maximum loss from a peak to a trough of a portfolio's value,
        expressed as a percentage.

        Returns:
          float: The maximum drawdown of the portfolio as a percentage.
        """
        cuml_ret = np.cumprod(1 + self.data)
        roll_max = np.maximum.accumulate(cuml_ret)
        drawdown = (cuml_ret - roll_max) / roll_max
        return np.min(drawdown)

    def portfolio_value_at_risk(self):
        """
      Calculates the portfolio value at risk (VaR) using the normal distribution.

      Returns:
        float: The portfolio value at risk.
      """
        return scipy.stats.norm.ppf(self.alph_var, np.mean(self.data),
                                    np.std(self.data))

    def portfolio_expected_loss(self):
        """
      Calculates the expected loss of the portfolio.

      Returns:
        float: The mean value of the data points that are below the portfolio's value at risk.
      """
        return self.data[self.data < self.portfolio_value_at_risk()].mean()

    def plot_cumulative_returns(self):
        """
      Plots the cumulative returns of the portfolio and the market index.

      This method calculates the cumulative returns of the portfolio and the market index,
      and then plots them on a line chart. The x-axis represents the dates, and the y-axis
      represents the cumulative returns.
      """
        if isinstance(self.mkt_ret, pd.Series):
            port_cum_ret = (1 + self.data).cumprod()
            mkt_cum_ret = (1 + self.mkt_ret).cumprod()
            plt.plot(port_cum_ret.index, port_cum_ret, label='Portfolio')
            plt.plot(mkt_cum_ret.index, mkt_cum_ret, label='Market Index')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.title('Portfolio vs. Market Index Cumulative Returns')
            plt.legend()
            plt.show()

    def __call__(self, verbose: bool = False):
        result = dict()
        result["annual_return"] = self.annualized_return(data=self.data)
        result["annual_std"] = self.annualized_std_return()
        result["beta"] = self.portfolio_beta()
        result["expected_return_capm"] = self.expected_return_capm()
        result["alpha"] = self.calculate_portfolio_alpha()
        result["sharpe_ratio"] = self.sharpe_ratio()
        result["max_drawdown"] = self.max_drawdown()
        var_txt = f"{self.period}_{(1-self.alph_var) * 100:.0f}"
        result[f"value_at_risk_{var_txt}"] = self.portfolio_value_at_risk()
        result[f"expected_loss_{var_txt}"] = self.portfolio_expected_loss()
        if verbose:
            self.plot_cumulative_returns()
        return result
