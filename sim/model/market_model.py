"""Market Model - the central simulation orchestrator.

This is the main Mesa Model that orchestrates the entire market simulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from mesa import Model
from mesa.time import Schedule

from ..exchange import MatchingEngine
from ..agents import TradingAgent, Strategy, get_loader
from ..news import NewsEvent
from .scheduler_logic import MarketEnvironment

if TYPE_CHECKING:
    from ..agents.strategy_interface import Observation


class MarketModel(Model):
    """Market simulation model - the central orchestrator.

    Responsibilities:
    - Simulation clock management
    - Tick orchestration (build observation, query agents, match, distribute fills)
    - Agent registry (add/remove agents, swap strategies)
    - Market environment state (fundamentals, noise, regimes)
    - Metrics collection via DataCollector
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        num_agents: int = 10,
        initial_cash: float = 10000.0,
        initial_price: float = 100.0,
        tick_interval: float = 1.0,
        agent_strategy: str = "random",
        agent_params: Optional[Dict] = None,
        enable_fundamentals: bool = False,
        enable_regimes: bool = False,
        regime_change_prob: float = 0.05,
        **kwargs,
    ):
        """Initialize the market model.

        Args:
            seed: Random seed for reproducibility
            num_agents: Number of trading agents
            initial_cash: Starting cash for each agent
            initial_price: Initial midprice
            tick_interval: Time between ticks
            agent_strategy: Default strategy for agents
            agent_params: Parameters for strategy
            **kwargs: Additional arguments for Mesa Model
        """
        super().__init__(seed=seed, **kwargs)

        # Simulation clock
        self._tick_count: float = 0.0
        self.tick_interval = tick_interval

        # Exchange
        self.exchange = MatchingEngine()

        # Agent scheduler - iterate over agents manually for Mesa 3.x
        # (No separate scheduler needed - use model's agents property)

        # Strategy loader
        self.strategy_loader = get_loader()

        # Market environment
        self.environment = MarketEnvironment(
            initial_price=initial_price,
            seed=seed,
            enable_fundamentals=enable_fundamentals,
            enable_regimes=enable_regimes,
            regime_change_prob=regime_change_prob,
        )

        # Agent registry
        self._agents: Dict[int, TradingAgent] = {}
        self._latest_news: Optional[NewsEvent] = None

        # Create agents
        agent_params = agent_params or {}
        for i in range(num_agents):
            # Create strategy
            strategy_kwargs = dict(agent_params)
            strategy_kwargs.setdefault("seed", self.random.randrange(2**32))
            strategy = self.strategy_loader.create(agent_strategy, **strategy_kwargs)

            # Create agent
            agent = TradingAgent(
                model=self,
                strategy=strategy,
                initial_cash=initial_cash,
            )
            self._agents[agent.unique_id] = agent

        # Seed the random number generator
        if seed is not None:
            self.reset_rng(seed)

        # DataCollector for metrics
        self._setup_data_collection()

        # Initial market state
        self._initialize_market(initial_price)

    def _setup_data_collection(self) -> None:
        """Set up Mesa DataCollector for metrics."""
        from ..metrics.datacollector_config import MetricsDataCollector

        self.datacollector = MetricsDataCollector(self)

    def _initialize_market(self, initial_price: float) -> None:
        """Initialize market with initial orders.

        Creates a simple order book to start with a spread.
        """
        from ..exchange import OrderType, Side

        if not self._agents:
            return

        # Add some initial orders to create a spread
        # This ensures there's always something to trade against

        # Create a few bid orders below initial price
        for i, price in enumerate(
            [initial_price - 0.5, initial_price - 1.0, initial_price - 1.5]
        ):
            agent = list(self._agents.values())[i % len(self._agents)]
            order = self.exchange.create_order(
                agent_id=agent.unique_id,
                side=Side.BID,
                order_type=OrderType.LIMIT,
                quantity=10.0,
                price=price,
            )
            self.exchange.submit_order(order)

        # Create a few ask orders above initial price
        for i, price in enumerate(
            [initial_price + 0.5, initial_price + 1.0, initial_price + 1.5]
        ):
            agent = list(self._agents.values())[(i + 3) % len(self._agents)]
            order = self.exchange.create_order(
                agent_id=agent.unique_id,
                side=Side.ASK,
                order_type=OrderType.LIMIT,
                quantity=10.0,
                price=price,
            )
            self.exchange.submit_order(order)

    @property
    def tick(self) -> float:
        """Current simulation tick."""
        return self._tick_count

    @property
    def agents(self) -> List[TradingAgent]:
        """Get all trading agents."""
        return list(self._agents.values())

    def step(self) -> None:
        """Execute one simulation step (tick).

        The tick orchestration is:
        1. Advance simulation clock
        2. Update market environment (fundamentals, noise)
        3. Build observation snapshot (handled by agents)
        4. Activate all agents to generate orders (via scheduler)
        5. Run matching (handled by exchange during order submission)
        6. Update agent states (handled during fill processing)
        7. Collect metrics
        8. Update visualization state
        """
        # Advance tick counter
        self._tick_count += 1
        self.exchange.next_tick()

        # Update market environment
        self.environment.update(self._tick_count, self.random)

        # Activate agents - iterate over all registered agents
        # Use self._agents.values() directly to avoid AgentSet issues
        for agent in list(self._agents.values()):
            agent.step()

        # Collect metrics after the step
        if self.datacollector is not None:
            self.datacollector.collect()

    def add_agent(
        self,
        strategy: Optional[Strategy] = None,
        initial_cash: float = 10000.0,
    ) -> TradingAgent:
        """Add a new trading agent.

        Args:
            strategy: Trading strategy (optional, uses default if None)
            initial_cash: Starting cash

        Returns:
            The created agent
        """
        # Use default strategy if none provided
        if strategy is None:
            strategy = self.strategy_loader.create(
                "random", seed=self.random.randrange(2**32)
            )

        # Create agent
        agent = TradingAgent(
            model=self,
            strategy=strategy,
            initial_cash=initial_cash,
        )

        self._agents[agent.unique_id] = agent

        return agent

    @property
    def current_news(self) -> Optional[NewsEvent]:
        """Most recently broadcast structured news event."""
        return self._latest_news

    def broadcast_news(self, news: Optional[NewsEvent | Dict]) -> Optional[NewsEvent]:
        """Broadcast a structured news event to all agents.

        Args:
            news: A `NewsEvent`, configuration dictionary, or `None` to clear

        Returns:
            The normalized broadcast event
        """
        event: Optional[NewsEvent]
        if isinstance(news, dict):
            event = NewsEvent.from_dict(news)
        else:
            event = news

        self._latest_news = event
        for agent in self._agents.values():
            agent.receive_news(event)

        return event

    def remove_agent(self, agent_id: int) -> Optional[TradingAgent]:
        """Remove an agent.

        Args:
            agent_id: ID of the agent to remove

        Returns:
            The removed agent, or None if not found
        """
        agent = self._agents.pop(agent_id, None)
        if agent is not None:
            for order in self.exchange.order_book.get_orders_for_agent(agent_id):
                self.exchange.order_book.remove_order(order.order_id)
            agent._pending_orders.clear()
            agent.remove()

        return agent

    def swap_strategy(self, agent_id: int, strategy: Strategy) -> bool:
        """Swap an agent's strategy (hot swap).

        Args:
            agent_id: ID of the agent
            strategy: New strategy instance

        Returns:
            True if successful, False if agent not found
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return False

        agent.set_strategy(strategy)
        return True

    def get_agent(self, agent_id: int) -> Optional[TradingAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_market_state(self) -> Dict:
        """Get comprehensive market state.

        Returns:
            Dictionary with market state
        """
        state = self.exchange.get_market_state()
        state["tick"] = self._tick_count
        state["environment"] = self.environment.get_state()
        state["num_agents"] = len(self._agents)
        state["news"] = self._latest_news.to_dict() if self._latest_news else None

        return state

    def get_agent_metrics(self) -> List[Dict]:
        """Get metrics for all agents.

        Returns:
            List of agent state dictionaries
        """
        return [agent.get_state() for agent in self._agents.values()]

    def get_leaderboard(self) -> List[Dict]:
        """Get agent leaderboard sorted by PnL.

        Returns:
            List of agents sorted by total_pnl
        """
        leaderboard = self.get_agent_metrics()
        leaderboard.sort(key=lambda x: x["total_pnl"], reverse=True)
        return leaderboard

    def reset(self, keep_agents: bool = True) -> None:
        """Reset the simulation.

        Args:
            keep_agents: If True, keep existing agents but reset their state
        """
        # Reset exchange
        self.exchange.reset()

        # Reset tick counter
        self._tick_count = 0.0

        # Reset environment
        self.environment.reset()
        self._latest_news = None

        if keep_agents:
            # Reset agent states
            for agent in self._agents.values():
                agent.cash = agent.initial_cash
                agent.position = 0.0
                agent._realized_pnl = 0.0
                agent._trade_history.clear()
                agent._filled_trades.clear()
                agent._pending_orders.clear()
                agent.receive_news(None)
                if agent.strategy is not None:
                    agent.strategy.reset()
        else:
            # Clear all agents
            self._agents.clear()
            # Re-register with model

    def __repr__(self) -> str:
        return (
            f"MarketModel(tick={self._tick_count}, "
            f"agents={len(self._agents)}, "
            f"best_bid={self.exchange.order_book.best_bid}, "
            f"best_ask={self.exchange.order_book.best_ask})"
        )
