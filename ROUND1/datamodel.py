"""
Stub local du datamodel Prosperity — pour tester en VS Code.
Sur la vraie plateforme, ce fichier est fourni automatiquement.
"""

from typing import Dict, List, Any


class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity  # positif = achat, négatif = vente

    def __repr__(self):
        side = "BUY " if self.quantity > 0 else "SELL"
        return f"{side} {abs(self.quantity)}x {self.symbol} @ {self.price}"


class OrderDepth:
    def __init__(self):
        # {prix: volume} — volumes positifs
        self.buy_orders: Dict[int, int] = {}
        # {prix: volume} — volumes négatifs (convention Prosperity)
        self.sell_orders: Dict[int, int] = {}


class TradingState:
    def __init__(
        self,
        traderData: str,
        timestamp: int,
        order_depths: Dict[str, OrderDepth],
        position: Dict[str, int],
    ):
        self.traderData = traderData
        self.timestamp = timestamp
        self.order_depths = order_depths
        self.position = position