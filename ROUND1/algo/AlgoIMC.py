import jsonpickle
from datamodel import OrderDepth, TradingState, Order
from typing import List

# ─────────────────────────────────────────────
#  POSITION LIMITS (officiels Round 1)
# ─────────────────────────────────────────────

POSITION_LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# OSMIUM : stable autour de 10 000 → mean reversion
OSMIUM_FAIR_VALUE = 10_000


class Trader:

    def run(self, state: TradingState):
        try:
            memory = jsonpickle.decode(state.traderData)
            if not isinstance(memory, dict):
                memory = {}
        except Exception:
            memory = {}

        result = {}
        conversions = 0

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            limit    = POSITION_LIMITS.get(product, 80)
            position = state.position.get(product, 0)
            buy_cap  = limit - position
            sell_cap = limit + position

            # ══════════════════════════════════════
            #  ASH_COATED_OSMIUM — Mean Reversion
            #  Prix stable autour de 10 000
            #  → Acheter tout sous 10 000
            #  → Vendre tout au-dessus de 10 000
            #  → Ordres passifs à ±1 pour le flux résiduel
            # ══════════════════════════════════════
            if product == "ASH_COATED_OSMIUM":
                fv = OSMIUM_FAIR_VALUE

                # Achat agressif : tout ask < 10 000
                if buy_cap > 0 and order_depth.sell_orders:
                    for ask_px in sorted(order_depth.sell_orders.keys()):
                        if ask_px >= fv or buy_cap <= 0:
                            break
                        qty = min(-order_depth.sell_orders[ask_px], buy_cap)
                        if qty > 0:
                            orders.append(Order(product, ask_px, qty))
                            buy_cap -= qty

                # Vente agressive : tout bid > 10 000
                if sell_cap > 0 and order_depth.buy_orders:
                    for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                        if bid_px <= fv or sell_cap <= 0:
                            break
                        qty = min(order_depth.buy_orders[bid_px], sell_cap)
                        if qty > 0:
                            orders.append(Order(product, bid_px, -qty))
                            sell_cap -= qty

                # Ordres passifs serrés pour flux résiduel
                if buy_cap > 0:
                    orders.append(Order(product, fv - 1, buy_cap))
                if sell_cap > 0:
                    orders.append(Order(product, fv + 1, -sell_cap))

            # ══════════════════════════════════════
            #  INTARIAN_PEPPER_ROOT — Buy and Hold
            #  Prix monte de +1 000 par jour (~+3 000 sur 3 jours)
            #  Avec limit=80 : gain théorique = 80 × 3000 = 240 000
            #  → Rester MAX LONG en permanence, ne jamais vendre
            # ══════════════════════════════════════
            elif product == "INTARIAN_PEPPER_ROOT":
                # Acheter tout ce qui est disponible
                if buy_cap > 0 and order_depth.sell_orders:
                    for ask_px in sorted(order_depth.sell_orders.keys()):
                        if buy_cap <= 0:
                            break
                        qty = min(-order_depth.sell_orders[ask_px], buy_cap)
                        if qty > 0:
                            orders.append(Order(product, ask_px, qty))
                            buy_cap -= qty

                # Ordre passif au meilleur ask pour capturer toute liquidité
                if buy_cap > 0 and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    orders.append(Order(product, best_ask, buy_cap))

            result[product] = orders

        trader_data = jsonpickle.encode(memory)
        return result, conversions, trader_data