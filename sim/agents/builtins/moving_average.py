"""Moving average strategy.

This module provides a moving average strategy that trades based on
price crossovers with a historical moving average.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent
from sim.agents.strategy_interface import Observation, OrderRequest, Strategy


class MovingAverageStrategy(Strategy):
    """Moving average trend-following strategy.

    This strategy maintains a rolling window of recent prices. It generates
    buy signals when the price crosses above the moving average and sell
    signals when the price crosses below it. It takes liquidity to capture
    the trend immediately.

    Attributes:
        window: Number of periods for the moving average
        order_quantity: Base size of each order
    """

    def __init__(
        self,
        window: int = 20,
        order_quantity: float = 10.0,
        seed: Optional[int] = None,
    ):
        """Initialize the moving average strategy.

        Args:
            window: Number of ticks to calculate the moving average
            order_quantity: Base size of each order
            seed: Random seed for determinism
        """
        self.window = window
        self.order_quantity = order_quantity
        self.price_history = deque(maxlen=window)
        self._random = random.Random(seed)
        self._latest_news: Optional[NewsEvent] = None

    def act(self, observation: Observation) -> List[OrderRequest]:
        """Generate moving average orders.

        Args:
            observation: Current market observation

        Returns:
            List of order requests
        """
        # Use reference_price as fallback when midprice is unavailable
        midprice = observation.midprice
        if midprice is None:
            midprice = observation.reference_price
        if midprice is None:
            return []

        # Record the current price
        self.price_history.append(midprice)

        # Wait until we have a full window of data before trading
        if len(self.price_history) < self.window:
            return []

        # Calculate the simple moving average
        sma = sum(self.price_history) / len(self.price_history)

        activity_multiplier = (
            self._latest_news.activity_multiplier() if self._latest_news else 1.0
        )
        adjusted_quantity = max(1.0, self.order_quantity * activity_multiplier)
        
        # Determine trade side based on MA crossover
        side = None
        if midprice > sma and observation.position <= 0:
            # Bullish crossover: price is above MA, and we aren't long yet
            side = Side.BID
        elif midprice < sma and observation.position >= 0:
            # Bearish crossover: price is below MA, and we aren't short yet
            side = Side.ASK
            
        # Optional: Let strong news override the technical signal
        directional_bias = (
            self._latest_news.directional_bias if self._latest_news else 0.0
        )
        if self._latest_news is not None and self._random.random() < min(
            1.0, abs(directional_bias)
        ):
            side = Side.BID if directional_bias >= 0 else Side.ASK

        # No signal, do nothing
        if side is None:
            return []

        # Calculate target quantity: close existing opposite position + establish new position
        target_qty = adjusted_quantity + abs(observation.position)

        # Safety check: ensure we have enough cash for bids
        if side == Side.BID and observation.cash < target_qty * midprice:
            target_qty = (observation.cash / midprice) * 0.95  # 5% buffer
            if target_qty < 1.0:
                return []

        # Submit a market order to take liquidity and ride the trend immediately
        return [
            OrderRequest(
                side=side,
                order_type=OrderType.MARKET,
                quantity=target_qty,
                price=None,  # Market orders don't need a price
            )
        ]

    def on_news(self, news: Optional[NewsEvent]) -> None:
        """Store the latest news for directional calibration."""
        self._latest_news = news

    def reset(self) -> None:
        """Reset strategy state and clear historical price data."""
        self._latest_news = None
        self.price_history.clear()

    def refresh_orders(self) -> bool:
        """Takers should not leave stale orders behind."""
        return True