"""Matching engine for the exchange.

This module implements the core matching logic that processes orders
and generates trades according to price-time priority.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .order import Order, OrderStatus, OrderType, Side, Trade
from .orderbook import OrderBook


class MatchingEngine:
    """Matching engine for a single instrument.

    The matching engine processes incoming orders against the order book,
    generating trades according to price-time priority rules.

    Attributes:
        order_book: The order book for this instrument
        tick: Current tick counter
    """

    def __init__(self) -> None:
        """Initialize the matching engine."""
        self.order_book = OrderBook()
        self.tick: float = 0.0
        self._trade_counter: int = 0
        self._order_counter: int = 0
        self._last_trades: List[Trade] = []

    def reset(self) -> None:
        """Reset the matching engine to initial state."""
        self.order_book.clear()
        self.tick = 0.0
        self._trade_counter = 0
        self._order_counter = 0
        self._last_trades.clear()

    def next_tick(self) -> None:
        """Advance to the next tick."""
        self.tick += 1
        # Clear last trades from previous tick
        self._last_trades.clear()

    def create_order(
        self,
        agent_id: int,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        """Create a new order with a unique ID.

        Args:
            agent_id: The agent placing the order
            side: BID or ASK
            order_type: LIMIT, MARKET, or CANCEL
            quantity: Order quantity
            price: Price for limit orders (None for market orders)

        Returns:
            The created order
        """
        self._order_counter += 1
        return Order(
            order_id=self._order_counter,
            agent_id=agent_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            timestamp=self.tick,
        )

    def submit_order(self, order: Order) -> OrderStatus:
        """Submit an order for matching.

        Args:
            order: The order to submit

        Returns:
            OrderStatus with fill information
        """
        if order.order_type == OrderType.CANCEL:
            return self._cancel_order(order)

        if order.order_type == OrderType.MARKET:
            return self._match_market_order(order)

        return self._match_limit_order(order)

    def _cancel_order(self, order: Order) -> OrderStatus:
        """Cancel an existing order."""
        existing = self.order_book.remove_order(order.order_id)
        if existing:
            return OrderStatus(
                order_id=order.order_id,
                filled_quantity=existing.filled_quantity,
                remaining_quantity=existing.remaining_quantity,
                canceled=True,
            )
        return OrderStatus(
            order_id=order.order_id,
            filled_quantity=0.0,
            remaining_quantity=0.0,
            canceled=True,
        )

    def _match_limit_order(self, order: Order) -> OrderStatus:
        """Match a limit order against the book.

        For a BUY (BID):
        - Match with lowest available ASK if price >= ask price
        - Remaining quantity goes on the book

        For a SELL (ASK):
        - Match with highest available BID if price <= bid price
        - Remaining quantity goes on the book
        """
        trades: List[Trade] = []
        filled_quantity = 0.0

        if order.side == Side.BID:
            # Buy order: match with asks
            while order.remaining_quantity > 0:
                best_ask = self.order_book.get_best_ask()
                if best_ask is None:
                    break

                # Check if we can match
                if order.price is not None and order.price < best_ask.price:
                    break

                # Execute match
                trade_qty = min(order.remaining_quantity, best_ask.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_ask,
                    taker_order=order,
                    price=best_ask.price,
                    quantity=trade_qty,
                )
                trades.append(trade)

                # Update quantities
                order.fill(trade_qty)
                best_ask.fill(trade_qty)
                filled_quantity += trade_qty

                # Clean up filled orders
                if best_ask.is_filled:
                    self.order_book.remove_order(best_ask.order_id)

        else:
            # Sell order: match with bids
            while order.remaining_quantity > 0:
                best_bid = self.order_book.get_best_bid()
                if best_bid is None:
                    break

                # Check if we can match
                if order.price is not None and order.price > best_bid.price:
                    break

                # Execute match
                trade_qty = min(order.remaining_quantity, best_bid.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_bid,
                    taker_order=order,
                    price=best_bid.price,
                    quantity=trade_qty,
                )
                trades.append(trade)

                # Update quantities
                order.fill(trade_qty)
                best_bid.fill(trade_qty)
                filled_quantity += trade_qty

                # Clean up filled orders
                if best_bid.is_filled:
                    self.order_book.remove_order(best_bid.order_id)

        # Add remaining quantity to book if not fully filled
        if not order.is_filled:
            self.order_book.add_order(order)

        self._last_trades.extend(trades)

        return OrderStatus(
            order_id=order.order_id,
            filled_quantity=filled_quantity,
            remaining_quantity=order.remaining_quantity,
            canceled=False,
            trades=trades,
        )

    def _match_market_order(self, order: Order) -> OrderStatus:
        """Match a market order against the book.

        Market orders are aggressive - they match at any price.
        They execute against the opposite side of the book.
        """
        trades: List[Trade] = []
        filled_quantity = 0.0

        if order.side == Side.BID:
            # Market buy: match with all asks
            while order.remaining_quantity > 0:
                best_ask = self.order_book.get_best_ask()
                if best_ask is None:
                    break

                # Execute match at ask price
                trade_qty = min(order.remaining_quantity, best_ask.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_ask,
                    taker_order=order,
                    price=best_ask.price,
                    quantity=trade_qty,
                )
                trades.append(trade)

                # Update quantities
                order.fill(trade_qty)
                best_ask.fill(trade_qty)
                filled_quantity += trade_qty

                # Clean up filled orders
                if best_ask.is_filled:
                    self.order_book.remove_order(best_ask.order_id)

        else:
            # Market sell: match with all bids
            while order.remaining_quantity > 0:
                best_bid = self.order_book.get_best_bid()
                if best_bid is None:
                    break

                # Execute match at bid price
                trade_qty = min(order.remaining_quantity, best_bid.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_bid,
                    taker_order=order,
                    price=best_bid.price,
                    quantity=trade_qty,
                )
                trades.append(trade)

                # Update quantities
                order.fill(trade_qty)
                best_bid.fill(trade_qty)
                filled_quantity += trade_qty

                # Clean up filled orders
                if best_bid.is_filled:
                    self.order_book.remove_order(best_bid.order_id)

        self._last_trades.extend(trades)

        return OrderStatus(
            order_id=order.order_id,
            filled_quantity=filled_quantity,
            remaining_quantity=order.remaining_quantity,
            canceled=False,
            trades=trades,
        )

    def _create_trade(
        self,
        maker_order: Order,
        taker_order: Order,
        price: float,
        quantity: float,
    ) -> Trade:
        """Create a trade record."""
        self._trade_counter += 1
        return Trade(
            trade_id=self._trade_counter,
            maker_order_id=maker_order.order_id,
            taker_order_id=taker_order.order_id,
            maker_agent_id=maker_order.agent_id,
            taker_agent_id=taker_order.agent_id,
            price=price,
            quantity=quantity,
            timestamp=self.tick,
            taker_side=taker_order.side,
        )

    def batch_match(self, orders: List[Order]) -> Dict[int, OrderStatus]:
        """Process a batch of orders.

        Args:
            orders: List of orders to process

        Returns:
            Dictionary mapping order_id to OrderStatus
        """
        results: Dict[int, OrderStatus] = {}

        # Process each order
        for order in orders:
            status = self.submit_order(order)
            results[order.order_id] = status

        return results

    def get_last_trades(self) -> List[Trade]:
        """Get the trades from the last matching round."""
        return self._last_trades.copy()

    def get_market_state(self) -> Dict:
        """Get current market state snapshot.

        Returns:
            Dictionary with market state information
        """
        return {
            "tick": self.tick,
            "best_bid": self.order_book.best_bid,
            "best_ask": self.order_book.best_ask,
            "midprice": self.order_book.midprice,
            "spread": self.order_book.spread,
            "volume": self.order_book.get_total_volume(),
            "bid_volume": self.order_book.get_total_volume(Side.BID),
            "ask_volume": self.order_book.get_total_volume(Side.ASK),
            "order_count": len(self.order_book),
            "last_trades": self._last_trades.copy(),
        }

    def get_depth_snapshot(self, depth: int = 10) -> Dict:
        """Get order book depth snapshot."""
        level2 = self.order_book.get_level2_snapshot(depth)
        return {
            "bids": level2["bids"],
            "asks": level2["asks"],
            "best_bid": self.order_book.best_bid,
            "best_ask": self.order_book.best_ask,
        }

    def __repr__(self) -> str:
        return (
            f"MatchingEngine(tick={self.tick}, "
            f"best_bid={self.order_book.best_bid}, "
            f"best_ask={self.order_book.best_ask}, "
            f"orders={len(self.order_book)})"
        )
