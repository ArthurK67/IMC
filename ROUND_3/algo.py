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
# Mean-reversion + passive market-making. Spread ~5, lag-1 autocorr ≈ -0.15.
VEF_SYMBOL = "VELVETFRUIT_EXTRACT"
VEF_POSITION_LIMIT = 200

# Slow EMA of midprice (~20-tick window); deviation drives directional targets
VEF_EMA_ALPHA = 0.05
VEF_TAKE_MAX_VOL = 30  # max units per level when taking on wide spread
VEF_TAKE_LIMIT = 170   # stop taking when |pos| would exceed this

# ── VEV options — per-strike EMA IV constants ─────────────────────────────────
SKIP_SYMBOLS = {"VEV_6000", "VEV_6500"}  # always return empty orders
VEV_POS_LIMIT = 300
INTERNAL_POS_LIMIT = 200     # self-imposed limit for aggressive taking only

# IV EMA: α=0.15 gives ~7-tick effective window
IV_EMA_ALPHA = 0.15

PASSIVE_VOLUME = 40
TAKE_VOLUME = 40
FLATTEN_THRESHOLD_VEV = 160  # abs(pos) above this triggers flatten
FLATTEN_TARGET_VEV = 80      # flatten aims to reduce abs(pos) to this level

TTE_DAYS = 5                 # days to expiry (round 3 → day 8 expiry, currently day 3)
VEV_T = TTE_DAYS / 365.0


# ── Black-Scholes helpers (module-level, pure Python) ─────────────────────────

def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0) -> float:
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return normal_cdf(d1)


def implied_vol(
    market_price: float, S: float, K: float, T: float, r: float = 0
) -> Optional[float]:
    intrinsic = max(0, S - K)
    if market_price <= intrinsic + 0.01:
        return None
    lo, hi = 0.001, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        price = bs_call_price(S, K, T, mid, r)
        if price < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.0001:
            break
    return (lo + hi) / 2


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


# ─────────────────────────────────────────────────────────────────────────────
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
        smile_state: dict = saved.get("vev_smile", {})

        # ── HYDROGEL_PACK: full logic ──────────────────────────────────────────
        result[SYMBOL] = self._trade_hydrogel(state, hp_state)

        # ── VELVETFRUIT_EXTRACT: full logic ───────────────────────────────────
        result[VEF_SYMBOL] = self._trade_velvetfruit(state, vef_state)

        # ── VEV options: IV-scalping ──────────────────────────────────────────
        vev_orders = self._trade_all_vev(state, smile_state)
        for sym, orders in vev_orders.items():
            result[sym] = orders

        # ── All remaining products: empty ─────────────────────────────────────
        active = {SYMBOL, VEF_SYMBOL} | set(vev_orders.keys())
        for sym in state.order_depths:
            if sym not in active:
                result[sym] = []

        # ── Persist state ──────────────────────────────────────────────────────
        saved[SYMBOL] = hp_state
        saved[VEF_SYMBOL] = vef_state
        saved["vev_smile"] = smile_state
        return result, 0, json.dumps(saved)

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, state: TradingState, s: dict) -> List[Order]:
        od: Optional[OrderDepth] = state.order_depths.get(SYMBOL)
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
        """
        Mean-reversion + spread-adaptive passive market-making for VELVETFRUIT_EXTRACT.

        Core idea: slow EMA tracks the fair price; deviations from it signal
        directional trades.  Passive quoting adapts to spread width and
        order-book imbalance.  When the spread is wide (≥5) we also take any
        order priced at or inside fair value.
        """
        od: Optional[OrderDepth] = state.order_depths.get(VEF_SYMBOL)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        # ── STEP 1: Book basics ────────────────────────────────────────────────
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        simple_mid = (best_bid + best_ask) / 2.0
        current_spread = best_ask - best_bid

        wall_bid = _max_vol_price(od.buy_orders)
        wall_ask = _max_vol_price(od.sell_orders)
        fair_value = (
            (wall_bid + wall_ask) / 2.0
            if (wall_bid is not None and wall_ask is not None)
            else simple_mid
        )

        # ── Bid-ask imbalance across the full book ─────────────────────────────
        total_bid_vol = sum(od.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total_vol = total_bid_vol + total_ask_vol
        imbalance = (total_bid_vol - total_ask_vol) / total_vol if total_vol > 0 else 0.0

        # ── STEP 2: Slow EMA → mean-reversion signal ──────────────────────────
        old_ema = s.get("ve_ema", simple_mid)
        ema_price = VEF_EMA_ALPHA * simple_mid + (1.0 - VEF_EMA_ALPHA) * old_ema
        s["ve_ema"] = ema_price
        deviation = simple_mid - ema_price

        # ── STEP 3: Target position based on deviation ────────────────────────
        if deviation > 4:
            target_pos = -150
        elif deviation > 2:
            target_pos = -80
        elif deviation < -4:
            target_pos = 150
        elif deviation < -2:
            target_pos = 80
        else:
            target_pos = 0

        pos = state.position.get(VEF_SYMBOL, 0)
        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        # ── STEP 4: Aggressive execution toward target ─────────────────────────
        if pos < target_pos:
            qty = min(30, target_pos - pos, VEF_POSITION_LIMIT - pos)
            if qty > 0:
                orders.append(Order(VEF_SYMBOL, best_ask, qty))
                buy_used += qty
        elif pos > target_pos:
            qty = min(30, pos - target_pos, VEF_POSITION_LIMIT + pos)
            if qty > 0:
                orders.append(Order(VEF_SYMBOL, best_bid, -qty))
                sell_used += qty

        # ── STEP 5: Wide-spread taking at fair value ──────────────────────────
        # When spread ≥ 5 there is enough room to take any order sitting at or
        # inside fair value — pure edge, independent of the directional signal.
        if current_spread >= 5:
            fv_round = round(fair_value)
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px > fv_round:
                    break
                can_buy = min(VEF_TAKE_MAX_VOL, VEF_TAKE_LIMIT - pos - buy_used)
                if can_buy <= 0:
                    break
                qty = min(abs(od.sell_orders[ask_px]), can_buy)
                if qty > 0:
                    orders.append(Order(VEF_SYMBOL, ask_px, qty))
                    buy_used += qty
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px < fv_round:
                    break
                can_sell = min(VEF_TAKE_MAX_VOL, VEF_TAKE_LIMIT + pos - sell_used)
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid_px], can_sell)
                if qty > 0:
                    orders.append(Order(VEF_SYMBOL, bid_px, -qty))
                    sell_used += qty

        # ── STEP 6: Spread-adaptive passive quoting ───────────────────────────
        passive_bid = best_bid + 1
        passive_ask = best_ask - 1

        if current_spread >= 3 and passive_bid < passive_ask:
            # Base volume by spread width
            base_vol = 10 if current_spread == 3 else 25   # ≥4 → 25

            # Imbalance-adjusted volumes: lean toward the stronger side
            if imbalance > 0.3:
                bid_vol = int(base_vol * 1.5)
                ask_vol = int(base_vol * 0.5)
            elif imbalance < -0.3:
                bid_vol = int(base_vol * 0.5)
                ask_vol = int(base_vol * 1.5)
            else:
                bid_vol = base_vol
                ask_vol = base_vol

            # Cap at remaining position capacity
            bid_vol = min(bid_vol, VEF_POSITION_LIMIT - pos - buy_used)
            ask_vol = min(ask_vol, VEF_POSITION_LIMIT + pos - sell_used)

            if bid_vol > 0:
                orders.append(Order(VEF_SYMBOL, passive_bid, bid_vol))
            if ask_vol > 0:
                orders.append(Order(VEF_SYMBOL, passive_ask, -ask_vol))

        return orders


    # ──────────────────────────────────────────────────────────────────────────
    def _trade_all_vev(
        self, state: TradingState, s: dict
    ) -> Dict[str, List[Order]]:
        """
        Per-strike EMA IV market-making for all VEV options.
        Each strike maintains its own smoothed implied volatility (α=0.03).
        VEV_6000 and VEV_6500 are always empty.
        Self-imposed position limit: ±INTERNAL_POS_LIMIT (100).
        """
        result: Dict[str, List[Order]] = {
            sym: []
            for sym in state.order_depths
            if sym.startswith("VEV_")
        }

        # ── STEP 1: Require underlying price ──────────────────────────────────
        vef_od: Optional[OrderDepth] = state.order_depths.get(VEF_SYMBOL)
        if not vef_od or not vef_od.buy_orders or not vef_od.sell_orders:
            return result
        S = (max(vef_od.buy_orders) + min(vef_od.sell_orders)) / 2.0
        T = VEV_T

        # ── STEP 2 + 3: Process each option ───────────────────────────────────
        for sym in list(result.keys()):
            if sym in SKIP_SYMBOLS:
                continue
            od: Optional[OrderDepth] = state.order_depths.get(sym)
            if not od or not od.buy_orders or not od.sell_orders:
                continue

            K = int(sym.split("_")[1])
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            midprice = (best_bid + best_ask) / 2.0
            intrinsic = max(0.0, S - K)

            # ── Determine theo_price and delta ────────────────────────────────
            if midprice <= intrinsic + 0.5:
                # No detectable time value
                if sym in ("VEV_4000", "VEV_4500"):
                    # Deep ITM: use intrinsic as fair value; skip EMA update
                    theo_price = S - K
                    delta = 1.0
                else:
                    continue   # no time value, can't price reliably → skip
            else:
                # Compute observed IV and update per-strike EMA
                obs_iv = implied_vol(midprice, S, K, T)
                if obs_iv is None or obs_iv < 0.01 or obs_iv > 3.0:
                    continue   # aberrant or missing IV → skip

                iv_key = f"iv_{sym}"
                old_iv = s.get(iv_key)
                ema_iv = (
                    obs_iv
                    if old_iv is None
                    else IV_EMA_ALPHA * obs_iv + (1.0 - IV_EMA_ALPHA) * old_iv
                )
                s[iv_key] = ema_iv

                theo_price = bs_call_price(S, K, T, ema_iv)
                delta = bs_delta(S, K, T, ema_iv)

            # ── STEP 3: Trade this option ──────────────────────────────────────
            pos = state.position.get(sym, 0)
            orders: List[Order] = []

            # A) Aggressive taking — trigger at 0.5% deviation (down from 1%)
            min_deviation = max(0.5, theo_price * 0.005)

            if best_ask < theo_price - min_deviation:
                qty = min(TAKE_VOLUME, INTERNAL_POS_LIMIT - pos)
                if qty > 0:
                    orders.append(Order(sym, best_ask, qty))

            if best_bid > theo_price + min_deviation:
                qty = min(TAKE_VOLUME, INTERNAL_POS_LIMIT + pos)
                if qty > 0:
                    orders.append(Order(sym, best_bid, -qty))

            # B) Passive market-making around theo_price
            bid_px = int(math.floor(theo_price)) - 1
            ask_px = int(math.ceil(theo_price)) + 1

            # Track volumes for multi-layer capacity accounting
            bid_vol = 0
            ask_vol = 0
            l2_bid  = 0
            l2_ask  = 0

            if bid_px < ask_px:
                # Delta-aware skew: deep-ITM (delta≈1) options get 2× the skew
                delta_multiplier = max(0.5, min(2.0, delta * 2))
                skew = int((pos / INTERNAL_POS_LIMIT) * 2 * delta_multiplier)
                bid_px -= skew
                ask_px -= skew

                # Full volume, hard-capped by VEV_POS_LIMIT (not INTERNAL_POS_LIMIT)
                bid_vol = min(PASSIVE_VOLUME, VEV_POS_LIMIT - pos)
                ask_vol = min(PASSIVE_VOLUME, VEV_POS_LIMIT + pos)

                if bid_vol > 0 and bid_px < theo_price:
                    orders.append(Order(sym, bid_px, bid_vol))
                if ask_vol > 0 and ask_px > theo_price:
                    orders.append(Order(sym, ask_px, -ask_vol))

                # Second passive layer — 2 ticks deeper
                second_bid = bid_px - 2
                second_ask = ask_px + 2
                rem_buy  = VEV_POS_LIMIT - pos - bid_vol
                rem_sell = VEV_POS_LIMIT + pos - ask_vol
                l2_bid = min(bid_vol, rem_buy)
                l2_ask = min(ask_vol, rem_sell)
                if l2_bid > 0 and second_bid < theo_price:
                    orders.append(Order(sym, second_bid, l2_bid))
                if l2_ask > 0 and second_ask > theo_price:
                    orders.append(Order(sym, second_ask, -l2_ask))

            # C) Overbid / undercut layer — front-of-queue inside market spread
            # Only add when the overbid/undercut is on the correct side of theo
            ob_cap = VEV_POS_LIMIT - pos - bid_vol - l2_bid
            uc_cap = VEV_POS_LIMIT + pos - ask_vol - l2_ask
            if best_bid + 1 < theo_price and ob_cap > 0:
                orders.append(Order(sym, best_bid + 1, min(20, ob_cap)))
            if best_ask - 1 > theo_price and uc_cap > 0:
                orders.append(Order(sym, best_ask - 1, -min(20, uc_cap)))

            # D) Flatten when position is extreme
            if abs(pos) > FLATTEN_THRESHOLD_VEV:
                flatten_px = int(round(theo_price))
                flatten_qty = min(10, abs(pos) - FLATTEN_TARGET_VEV)
                if flatten_qty > 0:
                    if pos > 0:
                        orders.append(Order(sym, flatten_px, -flatten_qty))
                    else:
                        orders.append(Order(sym, flatten_px, flatten_qty))

            result[sym] = orders

        return result
