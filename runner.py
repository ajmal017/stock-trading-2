import stock
import trader
import util
import agents
import evaluator

import threading
import numpy

EVALUATION_THRESHOLD = -0.5

class Runner:

    def __init__(self):
        self.init_equity = 75000
        self.alpaca = util.get_alpaca()
        self.stock_list = self.load_stock_list(util.get_stock_universe_list())
        self.eval_stock_list = self.load_stock_list(util.get_eval_stock_list())
        self.agent_list = self.create_agent_list(self.stock_list)

    def load_stock_list(self, tickers):
        stock_list = []

        for ticker in tickers:
            try:
                self.alpaca.get_barset(ticker, 'minute', limit=1000)[ticker]
                stock_list.append(stock.Stock(ticker, self.alpaca))
            except Exception as e:
                print("Caught Exception:", e)

        return stock_list


    def create_agent_list(self, stock_list):
        """ TODO: consider using a separate class to do this.
            Essentially, make it such that only one file needs to be changed
            when adding or removing agents.
        """
        ema_feature_set_factory = lambda s: (lambda: agents.feature_sets.EmaFeatureSet(s))
        bool_fib_feature_set_factory = agents.feature_sets.BooleanFibFeatureSet
        simple_fib_feature_set_factory = agents.feature_sets.SimpleFibFeatureSet
        time_series_feature_set_factory = lambda i: (lambda: agents.feature_sets.GrowthTimeSeriesFeatureSet(i))

        alligator = agents.williams_alligator.WilliamsAlligator("williams-alligator",
                        stock_list,
                        0.75)

        baseline_all_equity = agents.baseline.Baseline("baseline-1.0", stock_list, alpha=1.0)
        baseline_three_quarters_equity = agents.baseline.Baseline("baseline-0.75", stock_list, alpha=0.75)
        baseline_half_equity = agents.baseline.Baseline("baseline-0.5", stock_list, alpha=0.5)
        baseline_tenth_equity = agents.baseline.Baseline("baseline-0.1", stock_list, alpha=0.1)

        catch_rise = agents.catch_rise.CatchRise(
                        "catch-rise",
                        stock_list)

        rolling_bands_0_1 = agents.rolling_bands.RollingBands(
                        "rolling-bands-0.1",
                        stock_list,
                        deviation_multiplier=0.1)
        rolling_bands_0_5 = agents.rolling_bands.RollingBands(
                        "rolling-bands-0.5",
                        stock_list,
                        deviation_multiplier=0.5)
        rolling_bands_1_0 = agents.rolling_bands.RollingBands(
                        "rolling-bands-1.0",
                        stock_list,
                        deviation_multiplier=1.0)
        rolling_bands_1_5 = agents.rolling_bands.RollingBands(
                        "rolling-bands-1.5",
                        stock_list,
                        deviation_multiplier=1.5)
        rolling_bands_2_0 = agents.rolling_bands.RollingBands(
                        "rolling-bands-2.0",
                        stock_list,
                        deviation_multiplier=2.0)

        simple_ema_eps_agent = agents.ddqn_agent.DdqnAgent(
                        "simple-multiple-ema-epsilon-ddqn",
                        stock_list,
                        (ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)), ),
                        action_values=(0.0, 0.1, 0.5, 0.6, 0.9),
                        overall_alpha=1.0,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-7,
                        epsilon=0.05)

        simple_ema_bolt_agent = agents.ddqn_agent.DdqnAgent(
                        "simple-multiple-ema-boltzmann-ddqn",
                        stock_list,
                        (ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)), ),
                        action_values=(0.0, 0.1, 0.5, 0.6, 0.9),
                        overall_alpha=1.0,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-7,
                        init_temp=20.0,
                        cooldown_time=10)

        def _rolling_band_convert_action(agent, action, ticker, price):
            feature_set = agent.feature_generators[ticker].feature_sets[0]
            moving_avg = feature_set.mean[0]
            deviation = numpy.sqrt(feature_set.var[0]) * agent.action_values[int(action)]
            return min(1, max(0, (moving_avg + deviation - price) / (2 * deviation)))

        time_series_rolling_band_agent = agents.ddqn_agent.DdqnAgent(
                        "time-series-rolling-band-boltzmann-ddqn",
                        stock_list,
                        (ema_feature_set_factory((390, )), time_series_feature_set_factory(60)),
                        action_values=(0.1, 0.5, 1.0, 1.5, 2.0),
                        overall_alpha=1.0,
                        fc_layer_params=(256, 128),
                        alpha=5e-7,
                        epsilon=0.05,
                        convert_action=_rolling_band_convert_action)

        time_series_rolling_band_multi_layer_agent = agents.ddqn_agent.DdqnAgent(
                        "time-series-rolling-band-multi-layer-ddqn",
                        stock_list,
                        (ema_feature_set_factory((390, )), time_series_feature_set_factory(60)),
                        action_values=(0.1, 0.25, 0.5, 1.0, 1.5, 2.0),
                        overall_alpha=1.0,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-7,
                        epsilon=0.05,
                        convert_action=_rolling_band_convert_action)

        time_series_agent = agents.ddqn_agent.DdqnAgent(
                        "time-series-multi-layer-ddqn",
                        stock_list,
                        (time_series_feature_set_factory(60), ),
                        (0.0, 0.1, 0.5, 0.6, 0.9),
                        1.0,
                        (512, 256, 128),
                        1e-7)

        fib_bool_agent = agents.ddqn_agent.DdqnAgent(
                        "boolean-fib-ddqn",
                        stock_list,
                        (bool_fib_feature_set_factory, ),
                        (0.0, 0.1, 0.5, 0.75, 0.9),
                        1.0,
                        (512, 256, 128),
                        5e-7)

        return [rolling_bands_0_1,
                rolling_bands_0_5,
                rolling_bands_1_0,
                rolling_bands_1_5,
                rolling_bands_2_0,

                catch_rise,

                baseline_all_equity,
                baseline_three_quarters_equity,
                baseline_half_equity,
                baseline_tenth_equity,

                alligator,

                simple_ema_bolt_agent,
                simple_ema_eps_agent,

                time_series_rolling_band_agent,
                time_series_rolling_band_multi_layer_agent,

                time_series_agent,

                fib_bool_agent]

    def train_agents(self):
        threads = []

        def _train_and_evaluate_agent(agent):
            print("---- Training", agent.name)
            agent.train(training_iterations=30)
            print("---- Evaluating", agent.name)
            evaluator.EvalEnv(self.eval_stock_list, agent).run_and_save_evaluation()
            print("---- Done with", agent.name)

        for agent in self.agent_list:
            _train_and_evaluate_agent(agent)
            # thread = threading.Thread(target=lambda: _train_and_evaluate_agent(agent))
            # threads.append(thread)
            # thread.start()

        # for thread in threads:
        #     thread.join()

    def evaluate_agents(self):
        new_agent_list = []
        agent_evaluations = dict()

        for agent in self.agent_list:
            agent.load()
            evaluator.EvalEnv(self.eval_stock_list, agent).run_and_save_evaluation()
            agent_evaluation = agent.get_evaluation()
            agent_evaluations[agent.name] = dict()
            agent_evaluations[agent.name]["evaluation"] = agent_evaluation
            agent_evaluations[agent.name]["is_selected"] = agent_evaluation >= EVALUATION_THRESHOLD

            if agent_evaluations[agent.name]["is_selected"]:
                new_agent_list.append(agent)

        self.agent_list = new_agent_list

    def start_trading(self):
        self.evaluate_agents()
        t = trader.Trader(self.init_equity, self.stock_list, self.agent_list, self.alpaca)
        threading.Thread(target=t.start_live_trading).start()
