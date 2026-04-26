import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── HYDROGEL_PACK constants ────────────────────────────────────────────────────
SYMBOL = "HYDROGEL_PACK"
POSITION_LIMIT = 200

MAX_VOLUME = 40

# EMA pour mean reversion
HP_EMA_ALPHA = 0.02

# Mean reversion : seuils d'écart pour déclencher des prises agressives
MR_THRESHOLD_SMALL = 15
MR_THRESHOLD_LARGE = 30
MR_VOL_SMALL = 15
MR_VOL_LARGE = 35

# Aggressive-take parameters
TAKE_EDGE = 1
TAKE_LIMIT = 190
TAKE_MAX_VOL = 30

# Flatten parameters
FLATTEN_THRESHOLD = 150
FLATTEN_TARGET = 80

# ── VELVETFRUIT_EXTRACT constants ──────────────────────────────────────────────
VEF_SYMBOL = "VELVETFRUIT_EXTRACT"
VEF_POSITION_LIMIT = 200

VEF_HALF_SPREAD = 2
VEF_MAX_VOLUME = 20

VEF_TAKE_EDGE = 1
VEF_TAKE_LIMIT = 150
VEF_TAKE_MAX_VOL = 20

VEF_FLATTEN_THRESHOLD = 120
VEF_FLATTEN_TARGET = 60

# ── VEV deep-ITM options constants ────────────────────────────────────────────
VEV_ACTIVE_SYMBOLS = ("VEV_4000", "VEV_4500", "VEV_5000")
VEV_POSITION_LIMIT = 300

VEV_TAKE_EDGE = 1
VEV_TAKE_MAX_VOL = 30

VEV_MAX_VOLUME = 30

VEV_FLATTEN_THRESHOLD = 200
VEV_FLATTEN_TARGET = 100


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        try:
            saved: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        hp_state: dict = saved.get(SYMBOL, {})
        vef_state: dict = saved.get(VEF_SYMBOL, {})

        result[SYMBOL] = self._trade_hydrogel(state, hp_state)
        result[VEF_SYMBOL] = self._trade_velvetfruit(state, vef_state)

        for vev_sym in VEV_ACTIVE_SYMBOLS:
            vev_st: dict = saved.get(vev_sym, {})
            result[vev_sym] = self._trade_vev_mm(state, vev_st, vev_sym, VEV_POSITION_LIMIT)
            saved[vev_sym] = vev_st

        active = {SYMBOL, VEF_SYMBOL} | set(VEV_ACTIVE_SYMBOLS)
        for sym in state.order_depths:
            if sym not in active:
                result[sym] = []

        saved[SYMBOL] = hp_state
        saved[VEF_SYMBOL] = vef_state
        return result, 0, json.dumps(saved)

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, state: TradingState, s: dict) -> List[Order]:
        od: OrderDepth = state.order_depths.get(SYMBOL)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        for old_key in ("price_history", "vol_sum", "vol_count"):
            s.pop(old_key, None)

        # ── STEP 1: Fair value via Wall Mid ───────────────────────────────────
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        simple_mid = (best_bid + best_ask) / 2.0

        wall_bid = _max_vol_price(od.buy_orders)
        wall_ask = _max_vol_price(od.sell_orders)

        if wall_bid is not None and wall_ask is not None:
            current_mid = (wall_bid + wall_ask) / 2.0
        else:
            current_mid = simple_mid

        # EMA longue persistante — ancre de mean reversion
        # Initialisée à 10 000 au premier tick (valeur connue du produit)
        prev_ema = s.get("ema_long", 10000.0)
        ema_long = HP_EMA_ALPHA * current_mid + (1 - HP_EMA_ALPHA) * prev_ema
        s["ema_long"] = ema_long

        # Fair value instantanée pour le market making
        fair_value = current_mid
        s["ema_mid"] = fair_value

        pos = state.position.get(SYMBOL, 0)
        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        # ── STEP 2: Mean reversion agressive ─────────────────────────────────
        # Ecart entre prix actuel et EMA longue
        # Négatif = prix trop bas → acheter
        # Positif = prix trop haut → vendre
        deviation = current_mid - ema_long

        # Signal achat (prix trop bas vs moyenne)
        if deviation < -MR_THRESHOLD_LARGE:
            mr_buy_vol = MR_VOL_LARGE
        elif deviation < -MR_THRESHOLD_SMALL:
            mr_buy_vol = MR_VOL_SMALL
        else:
            mr_buy_vol = 0

        if mr_buy_vol > 0:
            for ask_px in sorted(od.sell_orders.keys()):
                if buy_used >= mr_buy_vol:
                    break
                if pos + buy_used >= TAKE_LIMIT:
                    break
                avail = abs(od.sell_orders[ask_px])
                qty = min(avail, mr_buy_vol - buy_used, TAKE_LIMIT - pos - buy_used)
                if qty > 0:
                    orders.append(Order(SYMBOL, ask_px, qty))
                    buy_used += qty

        # Signal vente (prix trop haut vs moyenne)
        if deviation > MR_THRESHOLD_LARGE:
            mr_sell_vol = MR_VOL_LARGE
        elif deviation > MR_THRESHOLD_SMALL:
            mr_sell_vol = MR_VOL_SMALL
        else:
            mr_sell_vol = 0

        if mr_sell_vol > 0:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if sell_used >= mr_sell_vol:
                    break
                if pos - sell_used <= -TAKE_LIMIT:
                    break
                avail = od.buy_orders[bid_px]
                qty = min(avail, mr_sell_vol - sell_used, TAKE_LIMIT + pos - sell_used)
                if qty > 0:
                    orders.append(Order(SYMBOL, bid_px, -qty))
                    sell_used += qty

        # ── STEP 3: Aggressive taking classique ──────────────────────────────
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px >= fair_value - TAKE_EDGE:
                break
            if pos + buy_used >= TAKE_LIMIT:
                break
            avail = abs(od.sell_orders[ask_px])
            qty = min(avail, TAKE_MAX_VOL, TAKE_LIMIT - pos - buy_used)
            if qty <= 0:
                break
            orders.append(Order(SYMBOL, ask_px, qty))
            buy_used += qty

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px <= fair_value + TAKE_EDGE:
                break
            if pos - sell_used <= -TAKE_LIMIT:
                break
            avail = od.buy_orders[bid_px]
            qty = min(avail, TAKE_MAX_VOL, TAKE_LIMIT + pos - sell_used)
            if qty <= 0:
                break
            orders.append(Order(SYMBOL, bid_px, -qty))
            sell_used += qty

        # ── STEP 4: Passive quoting ───────────────────────────────────────────
        bid_px = best_bid + 1
        ask_px = best_ask - 1

        if pos > 50:
            bid_px -= int(pos / 50)
        elif pos < -50:
            ask_px += int(abs(pos) / 50)

        max_bid = math.ceil(fair_value) - 1
        min_ask = math.floor(fair_value) + 1
        bid_px = min(bid_px, max_bid)
        ask_px = max(ask_px, min_ask)

        passive_bid_vol = 0
        passive_ask_vol = 0

        if bid_px < ask_px:
            rem_buy  = POSITION_LIMIT - pos - buy_used
            rem_sell = POSITION_LIMIT + pos - sell_used

            # Layer 1
            l1_bid = min(MAX_VOLUME, rem_buy)
            l1_ask = min(MAX_VOLUME, rem_sell)
            if l1_bid > 0:
                orders.append(Order(SYMBOL, bid_px, l1_bid))
                passive_bid_vol += l1_bid
            if l1_ask > 0:
                orders.append(Order(SYMBOL, ask_px, -l1_ask))
                passive_ask_vol += l1_ask

            # Layer 2
            second_bid = min(bid_px - 2, max_bid)
            second_ask = max(ask_px + 2, min_ask)
            if second_bid < second_ask:
                l2_bid = min(MAX_VOLUME, rem_buy - passive_bid_vol)
                l2_ask = min(MAX_VOLUME, rem_sell - passive_ask_vol)
                if l2_bid > 0:
                    orders.append(Order(SYMBOL, second_bid, l2_bid))
                    passive_bid_vol += l2_bid
                if l2_ask > 0:
                    orders.append(Order(SYMBOL, second_ask, -l2_ask))
                    passive_ask_vol += l2_ask

            # Layer 3 supprimé — la mean reversion couvre les gros écarts

        # ── STEP 5: Zero-edge flatten ─────────────────────────────────────────
        if pos > FLATTEN_THRESHOLD:
            flatten_px = round(fair_value)
            flatten_qty = min(10, pos - FLATTEN_TARGET)
            cap = POSITION_LIMIT + pos - sell_used - passive_ask_vol
            flatten_qty = min(flatten_qty, cap)
            if flatten_qty > 0:
                orders.append(Order(SYMBOL, flatten_px, -flatten_qty))

        elif pos < -FLATTEN_THRESHOLD:
            flatten_px = round(fair_value)
            flatten_qty = min(10, abs(pos) - FLATTEN_TARGET)
            cap = POSITION_LIMIT - pos - buy_used - passive_bid_vol
            flatten_qty = min(flatten_qty, cap)
            if flatten_qty > 0:
                orders.append(Order(SYMBOL, flatten_px, flatten_qty))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_velvetfruit(self, state: TradingState, s: dict) -> List[Order]:
        od: OrderDepth = state.order_depths.get(VEF_SYMBOL)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        simple_mid = (best_bid + best_ask) / 2.0

        wall_bid = _max_vol_price(od.buy_orders)
        wall_ask = _max_vol_price(od.sell_orders)

        fair_value = (wall_bid + wall_ask) / 2.0 if (wall_bid is not None and wall_ask is not None) else simple_mid
        s["ema_mid"] = fair_value

        pos = state.position.get(VEF_SYMBOL, 0)
        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px >= fair_value - VEF_TAKE_EDGE:
                break
            if pos + buy_used >= VEF_TAKE_LIMIT:
                break
            avail = abs(od.sell_orders[ask_px])
            qty = min(avail, VEF_TAKE_MAX_VOL, VEF_TAKE_LIMIT - pos - buy_used)
            if qty <= 0:
                break
            orders.append(Order(VEF_SYMBOL, ask_px, qty))
            buy_used += qty

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px <= fair_value + VEF_TAKE_EDGE:
                break
            if pos - sell_used <= -VEF_TAKE_LIMIT:
                break
            avail = od.buy_orders[bid_px]
            qty = min(avail, VEF_TAKE_MAX_VOL, VEF_TAKE_LIMIT + pos - sell_used)
            if qty <= 0:
                break
            orders.append(Order(VEF_SYMBOL, bid_px, -qty))
            sell_used += qty

        bid_px = round(fair_value) - VEF_HALF_SPREAD
        ask_px = round(fair_value) + VEF_HALF_SPREAD

        skew = int((pos / VEF_POSITION_LIMIT) * 3)
        bid_px -= skew
        ask_px -= skew

        max_bid = math.ceil(fair_value) - 1
        min_ask = math.floor(fair_value) + 1
        bid_px = min(bid_px, max_bid)
        ask_px = max(ask_px, min_ask)

        passive_bid_vol = 0
        passive_ask_vol = 0

        if bid_px < ask_px:
            rem_buy  = VEF_POSITION_LIMIT - pos - buy_used
            rem_sell = VEF_POSITION_LIMIT + pos - sell_used

            passive_bid_vol = min(VEF_MAX_VOLUME, rem_buy)
            passive_ask_vol = min(VEF_MAX_VOLUME, rem_sell)

            if passive_bid_vol > 0:
                orders.append(Order(VEF_SYMBOL, bid_px, passive_bid_vol))
            if passive_ask_vol > 0:
                orders.append(Order(VEF_SYMBOL, ask_px, -passive_ask_vol))

        if pos > VEF_FLATTEN_THRESHOLD:
            flatten_px = round(fair_value)
            flatten_qty = min(10, pos - VEF_FLATTEN_TARGET)
            cap = VEF_POSITION_LIMIT + pos - sell_used - passive_ask_vol
            flatten_qty = min(flatten_qty, cap)
            if flatten_qty > 0:
                orders.append(Order(VEF_SYMBOL, flatten_px, -flatten_qty))

        elif pos < -VEF_FLATTEN_THRESHOLD:
            flatten_px = round(fair_value)
            flatten_qty = min(10, abs(pos) - VEF_FLATTEN_TARGET)
            cap = VEF_POSITION_LIMIT - pos - buy_used - passive_bid_vol
            flatten_qty = min(flatten_qty, cap)
            if flatten_qty > 0:
                orders.append(Order(VEF_SYMBOL, flatten_px, flatten_qty))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_vev_mm(
        self, state: TradingState, s: dict, symbol: str, pos_limit: int
    ) -> List[Order]:
        od: OrderDepth = state.order_depths.get(symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        simple_mid = (best_bid + best_ask) / 2.0

        wall_bid = _max_vol_price(od.buy_orders)
        wall_ask = _max_vol_price(od.sell_orders)

        fair_value = (
            (wall_bid + wall_ask) / 2.0
            if (wall_bid is not None and wall_ask is not None)
            else simple_mid
        )
        s["ema_mid"] = fair_value

        pos = state.position.get(symbol, 0)
        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        take_limit = pos_limit - 50

        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px >= fair_value - VEV_TAKE_EDGE:
                break
            if pos + buy_used >= take_limit:
                break
            avail = abs(od.sell_orders[ask_px])
            qty = min(avail, VEV_TAKE_MAX_VOL, take_limit - pos - buy_used)
            if qty <= 0:
                break
            orders.append(Order(symbol, ask_px, qty))
            buy_used += qty

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px <= fair_value + VEV_TAKE_EDGE:
                break
            if pos - sell_used <= -take_limit:
                break
            avail = od.buy_orders[bid_px]
            qty = min(avail, VEV_TAKE_MAX_VOL, take_limit + pos - sell_used)
            if qty <= 0:
                break
            orders.append(Order(symbol, bid_px, -qty))
            sell_used += qty

        bid_px = best_bid + 1
        ask_px = best_ask - 1

        if pos > 50:
            bid_px -= int(pos / 50)
        elif pos < -50:
            ask_px += int(abs(pos) / 50)

        max_bid = math.ceil(fair_value) - 1
        min_ask = math.floor(fair_value) + 1
        bid_px = min(bid_px, max_bid)
        ask_px = max(ask_px, min_ask)

        passive_bid_vol = 0
        passive_ask_vol = 0

        if bid_px < ask_px:
            rem_buy  = pos_limit - pos - buy_used
            rem_sell = pos_limit + pos - sell_used

            passive_bid_vol = min(VEV_MAX_VOLUME, rem_buy)
            passive_ask_vol = min(VEV_MAX_VOLUME, rem_sell)

            if passive_bid_vol > 0:
                orders.append(Order(symbol, bid_px, passive_bid_vol))
            if passive_ask_vol > 0:
                orders.append(Order(symbol, ask_px, -passive_ask_vol))

        if abs(pos) > VEV_FLATTEN_THRESHOLD:
            flatten_px = round(fair_value)

            if pos > VEV_FLATTEN_THRESHOLD:
                flatten_qty = min(20, pos - VEV_FLATTEN_TARGET)
                cap = pos_limit + pos - sell_used - passive_ask_vol
                flatten_qty = min(flatten_qty, cap)
                if flatten_qty > 0:
                    orders.append(Order(symbol, flatten_px, -flatten_qty))
            else:
                flatten_qty = min(20, abs(pos) - VEV_FLATTEN_TARGET)
                cap = pos_limit - pos - buy_used - passive_bid_vol
                flatten_qty = min(flatten_qty, cap)
                if flatten_qty > 0:
                    orders.append(Order(symbol, flatten_px, flatten_qty))

        return orders


# ── Module-level helper ────────────────────────────────────────────────────────
def _max_vol_price(book: dict) -> Optional[float]:
    if not book:
        return None
    max_vol = max(abs(v) for v in book.values())
    candidates = [p for p, v in book.items() if abs(v) == max_vol]
    if len(candidates) != 1:
        return None
    return float(candidates[0])
