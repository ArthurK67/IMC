import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── HYDROGEL_PACK constants ────────────────────────────────────────────────────
SYMBOL = "HYDROGEL_PACK"
POSITION_LIMIT = 200

MAX_VOLUME = 40        # max units per passive order (both sides)

# Aggressive-take parameters
TAKE_EDGE = 1          # min ticks of edge required before taking (ask < fv-1, bid > fv+1)
TAKE_LIMIT = 190       # don't take if it would push |pos| past this (leave buffer)
TAKE_MAX_VOL = 30      # max units consumed per book level when taking

# Flatten-at-fair-value parameters
FLATTEN_THRESHOLD = 150   # abs(pos) above this triggers the zero-edge flatten order
FLATTEN_TARGET = 80       # flatten order targets |pos| = this value

# ── VELVETFRUIT_EXTRACT constants ──────────────────────────────────────────────
# Spread is ~5 ticks (vs ~16 for HYDROGEL), so quoting strategy differs.
VEF_SYMBOL = "VELVETFRUIT_EXTRACT"
VEF_POSITION_LIMIT = 200

VEF_HALF_SPREAD = 2    # quote at fair_value ± 2 → spread of 4, inside the ~5 market spread
VEF_MAX_VOLUME = 20

# Aggressive-take parameters (more conservative: leave room for VEV hedging later)
VEF_TAKE_EDGE = 1
VEF_TAKE_LIMIT = 150
VEF_TAKE_MAX_VOL = 20

# Flatten parameters
VEF_FLATTEN_THRESHOLD = 120
VEF_FLATTEN_TARGET = 60

# ── VEV deep-ITM options constants (shared across VEV_4000/4500/5000) ─────────
# These are delta-1 products with strong mean-reversion; same overbid/undercut
# approach as HYDROGEL_PACK but with a larger position limit (300).
VEV_ACTIVE_SYMBOLS = ("VEV_4000", "VEV_4500", "VEV_5000")
VEV_POSITION_LIMIT = 300

VEV_TAKE_EDGE = 1
VEV_TAKE_MAX_VOL = 30

VEV_MAX_VOLUME = 30

VEV_FLATTEN_THRESHOLD = 200   # abs(pos) above this triggers flatten
VEV_FLATTEN_TARGET = 100      # flatten order brings |pos| back to this level


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # ── Load persisted state ───────────────────────────────────────────────
        try:
            saved: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        hp_state: dict = saved.get(SYMBOL, {})
        vef_state: dict = saved.get(VEF_SYMBOL, {})

        # ── HYDROGEL_PACK: full logic ──────────────────────────────────────────
        result[SYMBOL] = self._trade_hydrogel(state, hp_state)

        # ── VELVETFRUIT_EXTRACT: full logic ───────────────────────────────────
        result[VEF_SYMBOL] = self._trade_velvetfruit(state, vef_state)

        # ── VEV deep-ITM options: generic market-making ───────────────────────
        for vev_sym in VEV_ACTIVE_SYMBOLS:
            vev_st: dict = saved.get(vev_sym, {})
            result[vev_sym] = self._trade_vev_mm(state, vev_st, vev_sym, VEV_POSITION_LIMIT)
            saved[vev_sym] = vev_st

        # ── All remaining products (VEV_5100+): empty ─────────────────────────
        active = {SYMBOL, VEF_SYMBOL} | set(VEV_ACTIVE_SYMBOLS)
        for sym in state.order_depths:
            if sym not in active:
                result[sym] = []

        # ── Persist state ──────────────────────────────────────────────────────
        saved[SYMBOL] = hp_state
        saved[VEF_SYMBOL] = vef_state
        return result, 0, json.dumps(saved)

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, state: TradingState, s: dict) -> List[Order]:
        od: OrderDepth = state.order_depths.get(SYMBOL)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        # ── Remove obsolete state keys from prior strategy versions ───────────
        for old_key in ("price_history", "vol_sum", "vol_count"):
            s.pop(old_key, None)

        # ── STEP 1: Fair value via Wall Mid ────────────────────────────────────
        # Wall mid anchors fair value at the book's largest-volume levels rather
        # than the inside spread.  Falls back to simple midprice on volume ties.
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        simple_mid = (best_bid + best_ask) / 2.0

        wall_bid = _max_vol_price(od.buy_orders)
        wall_ask = _max_vol_price(od.sell_orders)

        if wall_bid is not None and wall_ask is not None:
            fair_value = (wall_bid + wall_ask) / 2.0
        else:
            fair_value = simple_mid

        s["ema_mid"] = fair_value

        pos = state.position.get(SYMBOL, 0)
        orders: List[Order] = []
        buy_used = 0    # position capacity consumed by aggressive buys this tick
        sell_used = 0   # position capacity consumed by aggressive sells this tick

        # ── STEP 2: Aggressive taking (clear mispricings only) ─────────────────
        # Only take when there is at least TAKE_EDGE ticks of edge, and only up
        # to TAKE_LIMIT position to leave a buffer for passive quotes.

        # Buy any ask priced at least TAKE_EDGE below fair value (cheapest first)
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

        # Sell any bid priced at least TAKE_EDGE above fair value (highest first)
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

        # ── STEP 3: Passive quoting via overbid / undercut ─────────────────────
        # Base quotes: one tick inside the current best bid/ask.
        bid_px = best_bid + 1
        ask_px = best_ask - 1

        # Position skew: push the inventory-adding quote further away so we slow
        # down accumulation on the already-heavy side.
        # Only one side is skewed at a time (the side that would increase inventory).
        if pos > 50:
            bid_px -= int(pos / 50)     # lower bid when long
        elif pos < -50:
            ask_px += int(abs(pos) / 50)  # raise ask when short

        # Hard constraint: never buy at or above fair value, never sell at or
        # below it.  Prices are integers; largest int < fv and smallest int > fv:
        max_bid = math.ceil(fair_value) - 1   # e.g. fv=10000.5 → 10000
        min_ask = math.floor(fair_value) + 1  # e.g. fv=10000.5 → 10001

        bid_px = min(bid_px, max_bid)
        ask_px = max(ask_px, min_ask)

        # Passive orders across three layers.
        # Remaining capacity is shared across all layers; deeper layers are
        # skipped entirely if capacity is already exhausted.
        passive_bid_vol = 0   # total buy volume placed (used by Step 4 cap calc)
        passive_ask_vol = 0   # total sell volume placed

        if bid_px < ask_px:
            rem_buy = POSITION_LIMIT - pos - buy_used
            rem_sell = POSITION_LIMIT + pos - sell_used

            # Layer 1 — inside the spread (best_bid+1 / best_ask-1)
            l1_bid = min(MAX_VOLUME, rem_buy)
            l1_ask = min(MAX_VOLUME, rem_sell)
            if l1_bid > 0:
                orders.append(Order(SYMBOL, bid_px, l1_bid))
                passive_bid_vol += l1_bid
            if l1_ask > 0:
                orders.append(Order(SYMBOL, ask_px, -l1_ask))
                passive_ask_vol += l1_ask

            # Layer 2 — 2 ticks deeper than Layer 1
            second_bid = bid_px - 2
            second_ask = ask_px + 2
            second_bid = min(second_bid, max_bid)
            second_ask = max(second_ask, min_ask)
            if second_bid < second_ask:
                l2_bid = min(MAX_VOLUME, rem_buy - passive_bid_vol)
                l2_ask = min(MAX_VOLUME, rem_sell - passive_ask_vol)
                if l2_bid > 0:
                    orders.append(Order(SYMBOL, second_bid, l2_bid))
                    passive_bid_vol += l2_bid
                if l2_ask > 0:
                    orders.append(Order(SYMBOL, second_ask, -l2_ask))
                    passive_ask_vol += l2_ask

            # Layer 3 — 4 ticks deeper than Layer 1
            third_bid = bid_px - 4
            third_ask = ask_px + 4
            third_bid = min(third_bid, max_bid)
            third_ask = max(third_ask, min_ask)
            if third_bid < third_ask:
                l3_bid = min(MAX_VOLUME, rem_buy - passive_bid_vol)
                l3_ask = min(MAX_VOLUME, rem_sell - passive_ask_vol)
                if l3_bid > 0:
                    orders.append(Order(SYMBOL, third_bid, l3_bid))
                    passive_bid_vol += l3_bid
                if l3_ask > 0:
                    orders.append(Order(SYMBOL, third_ask, -l3_ask))
                    passive_ask_vol += l3_ask

        # ── STEP 4: Zero-edge flatten when position is extreme ─────────────────
        # Posts an extra order at fair value (zero spread, no edge collected) to
        # accelerate return toward ±FLATTEN_TARGET.
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

        # ── STEP 1: Fair value via Wall Mid ────────────────────────────────────
        # Identical logic to HYDROGEL_PACK — anchor to highest-volume level.
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

        # ── STEP 2: Aggressive taking ──────────────────────────────────────────
        # Take any edge ≥ 1 tick, up to VEF_TAKE_LIMIT to preserve capacity.

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

        # ── STEP 3: Passive quoting ────────────────────────────────────────────
        # The VEF spread is only ~5 ticks so overbid/undercut-by-1 would give
        # almost no edge.  Instead we quote at fair_value ± VEF_HALF_SPREAD (±2),
        # producing a 4-tick spread that sits inside the ~5-tick market spread.

        # Base quotes anchored to rounded fair value
        bid_px = round(fair_value) - VEF_HALF_SPREAD
        ask_px = round(fair_value) + VEF_HALF_SPREAD

        # Position skew: at pos=+200 skew=3, at pos=+100 skew=1; only one side
        # is skewed (the side that would worsen inventory).
        skew = int((pos / VEF_POSITION_LIMIT) * 3)
        bid_px -= skew
        ask_px -= skew

        # Hard constraint: integer prices strictly on the correct side of fair value
        max_bid = math.ceil(fair_value) - 1
        min_ask = math.floor(fair_value) + 1
        bid_px = min(bid_px, max_bid)
        ask_px = max(ask_px, min_ask)

        # Single passive layer (spread too tight for multiple layers)
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

        # ── STEP 4: Zero-edge flatten when position is extreme ─────────────────
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
        """
        Generic market-maker for deep-ITM VEV options.
        Uses overbid/undercut (same as HYDROGEL_PACK), single passive layer,
        and a flatten order when abs(pos) > VEV_FLATTEN_THRESHOLD.
        `pos_limit` is passed in so the method works for any VEV strike.
        """
        od: OrderDepth = state.order_depths.get(symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        # ── STEP 1: Fair value via Wall Mid ────────────────────────────────────
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

        # ── STEP 2: Aggressive taking ──────────────────────────────────────────
        # Stop taking at ±(pos_limit − 50) to preserve passive-quote capacity.
        take_limit = pos_limit - 50  # 250 when pos_limit = 300

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

        # ── STEP 3: Passive quoting via overbid / undercut ─────────────────────
        bid_px = best_bid + 1
        ask_px = best_ask - 1

        # One-sided position skew (only slow down the inventory-adding side)
        if pos > 50:
            bid_px -= int(pos / 50)
        elif pos < -50:
            ask_px += int(abs(pos) / 50)

        # Hard constraint: prices must be strictly on the correct side of fv
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

        # ── STEP 4: Zero-edge flatten when position is extreme ─────────────────
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


# ── Module-level helper (no state, easy to test) ───────────────────────────────
def _max_vol_price(book: dict) -> Optional[float]:
    """
    Return the price level in `book` that carries the highest absolute volume.
    Returns None on a volume tie so the caller falls back to simple midprice.
    Volumes may be negative on the sell side; abs() is applied throughout.
    """
    if not book:
        return None
    max_vol = max(abs(v) for v in book.values())
    candidates = [p for p, v in book.items() if abs(v) == max_vol]
    if len(candidates) != 1:
        return None
    return float(candidates[0])
