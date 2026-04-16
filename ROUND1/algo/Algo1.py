import jsonpickle
import numpy as np
import pandas as pd
import os
from datamodel import OrderDepth, TradingState, Order
from typing import List

# ─────────────────────────────────────────────
#  POSITION LIMITS (officiels Round 1)
# ─────────────────────────────────────────────

POSITION_LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

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

            if product == "ASH_COATED_OSMIUM":
                fv = OSMIUM_FAIR_VALUE

                if buy_cap > 0 and order_depth.sell_orders:
                    for ask_px in sorted(order_depth.sell_orders.keys()):
                        if ask_px >= fv or buy_cap <= 0:
                            break
                        qty = min(-order_depth.sell_orders[ask_px], buy_cap)
                        if qty > 0:
                            orders.append(Order(product, ask_px, qty))
                            buy_cap -= qty

                if sell_cap > 0 and order_depth.buy_orders:
                    for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                        if bid_px <= fv or sell_cap <= 0:
                            break
                        qty = min(order_depth.buy_orders[bid_px], sell_cap)
                        if qty > 0:
                            orders.append(Order(product, bid_px, -qty))
                            sell_cap -= qty

                if buy_cap > 0:
                    orders.append(Order(product, fv - 1, buy_cap))
                if sell_cap > 0:
                    orders.append(Order(product, fv + 1, -sell_cap))

            elif product == "INTARIAN_PEPPER_ROOT":
                if buy_cap > 0 and order_depth.sell_orders:
                    for ask_px in sorted(order_depth.sell_orders.keys()):
                        if buy_cap <= 0:
                            break
                        qty = min(-order_depth.sell_orders[ask_px], buy_cap)
                        if qty > 0:
                            orders.append(Order(product, ask_px, qty))
                            buy_cap -= qty

                if buy_cap > 0 and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    orders.append(Order(product, best_ask, buy_cap))

            result[product] = orders

        trader_data = jsonpickle.encode(memory)
        return result, conversions, trader_data


# ─────────────────────────────────────────────
#  BACKTESTER LOCAL
# ─────────────────────────────────────────────

def load_prices(folder: str) -> pd.DataFrame:
    frames = []
    for day in [-2, -1, 0]:
        path = os.path.join(folder, f"prices_round_1_day_{day}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, sep=";"))
    return pd.concat(frames).reset_index(drop=True)


def build_order_depth(row) -> OrderDepth:
    od = OrderDepth()
    for i in [1, 2, 3]:
        bp = row.get(f"bid_price_{i}"); bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}"); av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv): od.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av): od.sell_orders[int(ap)] = -int(av)
    return od


def simulate_fill(orders, order_depth, position, limit):
    cash_delta = 0; pos = position
    for order in orders:
        qty = order.quantity; price = order.price
        if qty > 0:
            for ask_px, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_px > price or qty <= 0: break
                fill = min(qty, -ask_vol, limit - pos)
                if fill <= 0: continue
                cash_delta -= fill * ask_px; pos += fill; qty -= fill
        else:
            for bid_px, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_px < price or qty >= 0: break
                fill = min(-qty, bid_vol, pos + limit)
                if fill <= 0: continue
                cash_delta += fill * bid_px; pos -= fill; qty += fill
    return cash_delta, pos


def run_backtest():
    folder = os.path.dirname(os.path.abspath(__file__))
    prices = load_prices(folder)
    prices = prices[prices["mid_price"] > 0]

    trader = Trader(); trader_data = ""
    positions = {p: 0 for p in POSITION_LIMITS}
    cash = {p: 0.0 for p in POSITION_LIMITS}
    pnl_history = []

    days = sorted(prices["day"].unique())
    print(f"\n{'='*65}")
    print(f"  BACKTEST FINAL — limit=80, {len(days)} jours")
    print(f"{'='*65}\n")

    for day in days:
        day_prices = prices[prices["day"] == day]
        for ts in sorted(day_prices["timestamp"].unique()):
            tick_rows = day_prices[day_prices["timestamp"] == ts]
            order_depths = {}; mid_prices = {}
            for _, row in tick_rows.iterrows():
                prod = row["product"]
                if prod in POSITION_LIMITS:
                    order_depths[prod] = build_order_depth(row)
                    mid_prices[prod] = float(row["mid_price"])

            state = TradingState(trader_data, int(ts), order_depths, dict(positions))
            result, _, trader_data = trader.run(state)

            for prod, orders in result.items():
                if prod not in order_depths: continue
                dc, new_pos = simulate_fill(orders, order_depths[prod], positions[prod], POSITION_LIMITS[prod])
                cash[prod] += dc; positions[prod] = new_pos

            total_pnl = sum(cash[p] + positions[p] * mid_prices.get(p, 0) for p in POSITION_LIMITS)
            pnl_history.append({"day": day, "timestamp": ts, "total_pnl": total_pnl})

        last_mids = {p: float(prices[(prices["product"]==p)&(prices["day"]==day)]["mid_price"].iloc[-1]) for p in POSITION_LIMITS}
        total_pnl = sum(cash[p] + positions[p] * last_mids[p] for p in POSITION_LIMITS)
        print(f"  Fin jour {day:+d} | "
              + " | ".join(f"{p[:7]}: pos={positions[p]:+3d}  mtm={positions[p]*last_mids[p]:+9.0f}" for p in POSITION_LIMITS)
              + f"  |  PnL = {total_pnl:+.0f}")

    hist = pd.DataFrame(pnl_history)
    final_pnl = hist["total_pnl"].iloc[-1]
    print(f"\n{'='*65}")
    print(f"  PnL FINAL : {final_pnl:+.2f} XIRECS")
    print(f"  OBJECTIF  : 200 000 XIRECS")
    print(f"  RÉSULTAT  : {'✅ ATTEINT' if final_pnl >= 200000 else '❌ NON ATTEINT'}")
    print(f"{'='*65}\n")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 5))
    plt.plot(hist["total_pnl"].values, color="steelblue", linewidth=1)
    plt.axhline(200000, color="red", linewidth=1, linestyle="--", label="Objectif 200k")
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.title("PnL cumulé — Round 1 Final (limit=80)")
    plt.ylabel("PnL (XIRECS)"); plt.xlabel("Tick")
    plt.legend(); plt.grid(True); plt.tight_layout()
    out_png = os.path.join(folder, "pnl_final.png")
    plt.savefig(out_png); plt.show()
    print(f"  Courbe sauvegardée : {out_png}")


if __name__ == "__main__":
    run_backtest()