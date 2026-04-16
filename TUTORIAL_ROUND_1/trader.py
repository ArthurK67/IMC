from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np

# ============================================================
#  CONFIGURATION
# ============================================================

# Limites de position (à adapter selon les règles de la compétition)
POSITION_LIMITS = {
    "TOMATOES": 20,
    "EMERALDS": 20,
}

# Paramètres EMERALDS
EMERALDS_FAIR_VALUE = 10000
EMERALDS_SPREAD     = 3        # on se place à ±3 autour du fair value (inside le spread de 16)

# Paramètres TOMATOES
TOMATOES_MA_WINDOW  = 20       # nombre de mid_prices gardées en mémoire pour la moyenne mobile
TOMATOES_SPREAD     = 4        # demi-spread autour du fair value calculé
TOMATOES_AGGR_THRESH = 8       # si mid_price s'écarte de + de X du fair value → ordre agressif


# ============================================================
#  TRADER
# ============================================================

class Trader:

    def run(self, state: TradingState):
        """
        Appelée à chaque timestamp par le moteur de simulation.
        Retourne un dict {symbol: [Order, ...]} et un string (mémoire persistante).
        """

        # --- Charger la mémoire du tick précédent ---
        memory = {}
        if state.traderData and state.traderData != "":
            try:
                memory = jsonpickle.decode(state.traderData)
            except Exception:
                memory = {}

        # Initialiser l'historique des mid_prices si absent
        if "tomatoes_prices" not in memory:
            memory["tomatoes_prices"] = []

        result = {}

        for product, order_depth in state.order_depths.items():

            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 20)

            # ── Calcul du mid_price courant ──────────────────────────────
            best_bid = max(order_depth.buy_orders.keys())  if order_depth.buy_orders  else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is None or best_ask is None:
                continue  # pas de liquidité, on passe

            mid_price = (best_bid + best_ask) / 2

            # ════════════════════════════════════════════════════════════
            #  EMERALDS  —  Market Making pur autour de 10 000
            # ════════════════════════════════════════════════════════════
            if product == "EMERALDS":

                fair_value = EMERALDS_FAIR_VALUE

                # ---------- Ordres passifs (market making) ----------
                # BID : on achète si quelqu'un veut vendre en-dessous du fair value
                bid_price = fair_value - EMERALDS_SPREAD
                ask_price = fair_value + EMERALDS_SPREAD

                buy_capacity  = limit - position        # combien on peut encore acheter
                sell_capacity = limit + position        # combien on peut encore vendre

                if buy_capacity > 0:
                    orders.append(Order(product, bid_price, buy_capacity))

                if sell_capacity > 0:
                    orders.append(Order(product, ask_price, -sell_capacity))

                # ---------- Ordres agressifs (mean reversion) ----------
                # Si quelqu'un vend EN DESSOUS du fair value → on achète tout de suite
                for ask, vol in sorted(order_depth.sell_orders.items()):
                    if ask < fair_value and buy_capacity > 0:
                        qty = min(-vol, buy_capacity)   # sell_orders stockés en négatif
                        orders.append(Order(product, ask, qty))
                        buy_capacity -= qty

                # Si quelqu'un achète AU DESSUS du fair value → on vend tout de suite
                for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid > fair_value and sell_capacity > 0:
                        qty = min(vol, sell_capacity)
                        orders.append(Order(product, bid, -qty))
                        sell_capacity -= qty

            # ════════════════════════════════════════════════════════════
            #  TOMATOES  —  Market Making + Mean Reversion sur MA mobile
            # ════════════════════════════════════════════════════════════
            elif product == "TOMATOES":

                # Mettre à jour l'historique
                memory["tomatoes_prices"].append(mid_price)
                if len(memory["tomatoes_prices"]) > TOMATOES_MA_WINDOW:
                    memory["tomatoes_prices"].pop(0)

                # Fair value = moyenne mobile (ou mid_price si pas assez de données)
                fair_value = float(np.mean(memory["tomatoes_prices"]))

                buy_capacity  = limit - position
                sell_capacity = limit + position

                # ---------- Ordres passifs (market making) ----------
                bid_price = round(fair_value - TOMATOES_SPREAD)
                ask_price = round(fair_value + TOMATOES_SPREAD)

                if buy_capacity > 0:
                    orders.append(Order(product, bid_price, buy_capacity))

                if sell_capacity > 0:
                    orders.append(Order(product, ask_price, -sell_capacity))

                # ---------- Ordres agressifs (mean reversion forte) ----------
                # Si le prix ask est très bas par rapport au fair value → on snipe
                for ask, vol in sorted(order_depth.sell_orders.items()):
                    if ask < fair_value - TOMATOES_AGGR_THRESH and buy_capacity > 0:
                        qty = min(-vol, buy_capacity)
                        orders.append(Order(product, ask, qty))
                        buy_capacity -= qty

                # Si le prix bid est très haut par rapport au fair value → on vend
                for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid > fair_value + TOMATOES_AGGR_THRESH and sell_capacity > 0:
                        qty = min(vol, sell_capacity)
                        orders.append(Order(product, bid, -qty))
                        sell_capacity -= qty

            result[product] = orders

        # --- Sauvegarder la mémoire pour le prochain tick ---
        trader_data = jsonpickle.encode(memory)

        # conversions (optionnel, non utilisé ici)
        conversions = 0

        return result, conversions, trader_data
