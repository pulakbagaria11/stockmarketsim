"""Exponential Moving Average strategy.

This module provides an EMA strategy that trades based on
price crossovers with a weighted, exponentially smoothed average.
"""

from __future__ import annotations

import random
from typing import List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent
from sim.agents.strategy_interface import Observation, OrderRequest, Strategy


class EMAStrategy(Strategy):
    """Exponential Moving Average trend-following strategy.

    This strategy maintains an exponentially smoothed moving average. 
    It generates buy signals when the price crosses above the EMA and sell
    signals when the price crosses below it. It takes liquidity to capture
    the trend immediately.

    Attributes:
        window: Number of periods for the baseline EMA calculation
        order_quantity: Base size of each order
    """

    def __init__(
        self,
        window: int = 20,
        order_quantity: float = 10.0,
        seed: Optional[int] = None,
    ):
        """Initialize the EMA strategy.

        Args:
            window: Number of ticks to calculate the baseline smoothing factor
            order_quantity: Base size of each order
            seed: Random seed for determinism
        """
        self.window = window
        self.order_quantity = order_quantity
        # Calculate the EMA smoothing constant (alpha)
        self.alpha = 2.0 / (window + 1.0)
        
        self.current_ema: Optional[float] = None
        self.initial_prices: List[float] = []
        
        self._random = random.Random(seed)
        self._latest_news: Optional[NewsEvent] = None

    def act(self, observation: Observation) -> List[OrderRequest]:
        """Generate EMA orders.

        Args:
            observation: Current market observation

        Returns:
            List of order requests
        """
        midprice = observation.midprice
        if midprice is None:
            midprice = observation.reference_price
        if midprice is None:
            return []

        # Initialize the EMA if it hasn't been set yet
        if self.current_ema is None:
            self.initial_prices.append(midprice)
            # Wait until we have enough data points to create a reliable baseline SMA
            if len(self.initial_prices) < self.window:
                return []
            # First EMA is simply the SMA of the initial window
            self.current_ema = sum(self.initial_prices) / self.window
        else:
            # Update the EMA with the new price
            self.current_ema = (midprice - self.current_ema) * self.alpha + self.current_ema

        activity_multiplier = (
            self._latest_news.activity_multiplier() if self._latest_news else 1.0
        )
        adjusted_quantity = max(1.0, self.order_quantity * activity_multiplier)
        
        # Determine trade side based on EMA crossover
        side = None
        if midprice > self.current_ema and observation.position <= 0:
            # Bullish crossover: price is above EMA, and we aren't long yet
            side = Side.BID
        elif midprice < self.current_ema and observation.position >= 0:
            # Bearish crossover: price is below EMA, and we aren't short yet
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

        # Submit a market order
        return [
            OrderRequest(
                side=side,
                order_type=OrderType.MARKET,
                quantity=target_qty,
                price=None,
            )
        ]

    def on_news(self, news: Optional[NewsEvent]) -> None:
        """Store the latest news for directional calibration."""
        self._latest_news = news

    def reset(self) -> None:
        """Reset strategy state."""
        self._latest_news = None
        self.current_ema = None
        self.initial_prices.clear()

    def refresh_orders(self) -> bool:
        """Takers should not leave stale orders behind."""
        return True