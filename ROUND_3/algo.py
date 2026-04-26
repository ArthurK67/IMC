import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── HYDROGEL_PACK constants ────────────────────────────────────────────────────
SYMBOL = "HYDROGEL_PACK"
POSITION_LIMIT = 200
MAX_VOLUME = 50            # Change 11: 40 → 50

HP_EMA_ALPHA = 0.02        # slow EMA for mean-reversion anchor
MR_THRESHOLD_SMALL = 15
MR_THRESHOLD_LARGE = 30
MR_VOL_SMALL = 15
MR_VOL_LARGE = 35

TAKE_EDGE = 1
TAKE_LIMIT = 190
TAKE_MAX_VOL = 40          # Change 11: 30 → 40

FLATTEN_THRESHOLD = 150
FLATTEN_TARGET = 80

# ── VELVETFRUIT_EXTRACT constants ──────────────────────────────────────────────
VEF_SYMBOL = "VELVETFRUIT_EXTRACT"
VEF_POSITION_LIMIT = 200

VEF_TAKE_LIMIT = 190       # Change 8: 150 → 190
VEF_TAKE_MAX_VOL = 40      # Change 8: 20 → 40
VEF_PASSIVE_VOL = 30       # Change 7: overbid/undercut volume per layer

VEF_FLATTEN_THRESHOLD = 120
VEF_FLATTEN_TARGET = 60

# ── VEV options — per-strike EMA IV constants ─────────────────────────────────
SKIP_SYMBOLS = {"VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}  # Change 1
VEV_POS_LIMIT = 300
INTERNAL_POS_LIMIT = 250   # Change 9: 150 → 250

IV_EMA_ALPHA = 0.1

PASSIVE_VOLUME = 40        # Change 9: 25 → 40
TAKE_VOLUME = 40           # Change 9: 25 → 40
FLATTEN_THRESHOLD_VEV = 120
FLATTEN_TARGET_VEV = 50

TTE_DAYS = 5
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
    """Return price with highest absolute volume; None on tie or empty book."""
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

        try:
            saved: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        hp_state: dict = saved.get(SYMBOL, {})
        vef_state: dict = saved.get(VEF_SYMBOL, {})
        smile_state: dict = saved.get("vev_smile", {})

        result[SYMBOL] = self._trade_hydrogel(state, hp_state)
        result[VEF_SYMBOL] = self._trade_velvetfruit(state, vef_state)

        vev_orders = self._trade_all_vev(state, smile_state)
        for sym, orders in vev_orders.items():
            result[sym] = orders

        active = {SYMBOL, VEF_SYMBOL} | set(vev_orders.keys())
        for sym in state.order_depths:
            if sym not in active:
                result[sym] = []

        saved[SYMBOL] = hp_state
        saved[VEF_SYMBOL] = vef_state
        saved["vev_smile"] = smile_state
        return result, 0, json.dumps(saved)

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, state: TradingState, s: dict) -> List[Order]:
        """
        Teammate's mean-reversion market-maker for HYDROGEL_PACK.
        Wall Mid → long EMA → MR aggressive → classic taking → overbid/undercut
        (3 layers) → flatten → EOD aggressive flatten.
        """
        od: Optional[OrderDepth] = state.order_depths.get(SYMBOL)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        for old_key in ("price_history", "vol_sum", "vol_count"):
            s.pop(old_key, None)

        # ── STEP 1: Wall Mid ──────────────────────────────────────────────────
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        simple_mid = (best_bid + best_ask) / 2.0

        wall_bid = _max_vol_price(od.buy_orders)
        wall_ask = _max_vol_price(od.sell_orders)
        current_mid = (
            (wall_bid + wall_ask) / 2.0
            if (wall_bid is not None and wall_ask is not None)
            else simple_mid
        )

        # ── STEP 2: Persistent slow EMA (mean-reversion anchor) ───────────────
        prev_ema = s.get("ema_long", 10000.0)
        ema_long = HP_EMA_ALPHA * current_mid + (1 - HP_EMA_ALPHA) * prev_ema
        s["ema_long"] = ema_long

        fair_value = current_mid
        s["ema_mid"] = fair_value

        pos = state.position.get(SYMBOL, 0)
        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        # ── STEP 3: Mean-reversion aggressive takes ────────────────────────────
        deviation = current_mid - ema_long

        mr_buy_vol = (
            MR_VOL_LARGE if deviation < -MR_THRESHOLD_LARGE else
            MR_VOL_SMALL if deviation < -MR_THRESHOLD_SMALL else 0
        )
        if mr_buy_vol > 0:
            for ask_px in sorted(od.sell_orders.keys()):
                if buy_used >= mr_buy_vol or pos + buy_used >= TAKE_LIMIT:
                    break
                avail = abs(od.sell_orders[ask_px])
                qty = min(avail, mr_buy_vol - buy_used, TAKE_LIMIT - pos - buy_used)
                if qty > 0:
                    orders.append(Order(SYMBOL, ask_px, qty))
                    buy_used += qty

        mr_sell_vol = (
            MR_VOL_LARGE if deviation > MR_THRESHOLD_LARGE else
            MR_VOL_SMALL if deviation > MR_THRESHOLD_SMALL else 0
        )
        if mr_sell_vol > 0:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if sell_used >= mr_sell_vol or pos - sell_used <= -TAKE_LIMIT:
                    break
                avail = od.buy_orders[bid_px]
                qty = min(avail, mr_sell_vol - sell_used, TAKE_LIMIT + pos - sell_used)
                if qty > 0:
                    orders.append(Order(SYMBOL, bid_px, -qty))
                    sell_used += qty

        # ── STEP 4: Classic edge taking ───────────────────────────────────────
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px >= fair_value - TAKE_EDGE or pos + buy_used >= TAKE_LIMIT:
                break
            avail = abs(od.sell_orders[ask_px])
            qty = min(avail, TAKE_MAX_VOL, TAKE_LIMIT - pos - buy_used)
            if qty <= 0:
                break
            orders.append(Order(SYMBOL, ask_px, qty))
            buy_used += qty

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px <= fair_value + TAKE_EDGE or pos - sell_used <= -TAKE_LIMIT:
                break
            avail = od.buy_orders[bid_px]
            qty = min(avail, TAKE_MAX_VOL, TAKE_LIMIT + pos - sell_used)
            if qty <= 0:
                break
            orders.append(Order(SYMBOL, bid_px, -qty))
            sell_used += qty

        # ── STEP 5: Passive overbid / undercut (3 layers) ─────────────────────
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

            l1_bid = min(MAX_VOLUME, rem_buy)
            l1_ask = min(MAX_VOLUME, rem_sell)
            if l1_bid > 0:
                orders.append(Order(SYMBOL, bid_px, l1_bid))
                passive_bid_vol += l1_bid
            if l1_ask > 0:
                orders.append(Order(SYMBOL, ask_px, -l1_ask))
                passive_ask_vol += l1_ask

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

            # Layer 3 — Change 11: restored
            third_bid = min(bid_px - 4, max_bid)
            third_ask = max(ask_px + 4, min_ask)
            if third_bid < third_ask:
                l3_bid = min(MAX_VOLUME, rem_buy - passive_bid_vol)
                l3_ask = min(MAX_VOLUME, rem_sell - passive_ask_vol)
                if l3_bid > 0:
                    orders.append(Order(SYMBOL, third_bid, l3_bid))
                    passive_bid_vol += l3_bid
                if l3_ask > 0:
                    orders.append(Order(SYMBOL, third_ask, -l3_ask))
                    passive_ask_vol += l3_ask

        # ── STEP 5b: Zero-edge flatten ─────────────────────────────────────────
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

        # ── STEP 6: End-of-day aggressive flatten — Change 3 ──────────────────
        if state.timestamp > 950000:
            if pos > 0:
                eod_qty = min(pos, POSITION_LIMIT + pos)
                if eod_qty > 0:
                    orders.append(Order(SYMBOL, best_bid, -eod_qty))
            elif pos < 0:
                eod_qty = min(abs(pos), POSITION_LIMIT - pos)
                if eod_qty > 0:
                    orders.append(Order(SYMBOL, best_ask, eod_qty))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_velvetfruit(self, state: TradingState, s: dict) -> List[Order]:
        """
        Simple passive market-maker for VELVETFRUIT_EXTRACT.
        Wall Mid → aggressive take → wide-spread take at FV → overbid/undercut
        2 layers → flatten → EOD flatten.
        """
        od: Optional[OrderDepth] = state.order_depths.get(VEF_SYMBOL)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        # ── A) Wall Mid fair value ─────────────────────────────────────────────
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

        pos = state.position.get(VEF_SYMBOL, 0)
        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        # ── B) Aggressive taking (edge ≥ 1 tick) ──────────────────────────────
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px >= fair_value - 1 or pos + buy_used >= VEF_TAKE_LIMIT:
                break
            avail = abs(od.sell_orders[ask_px])
            qty = min(avail, VEF_TAKE_MAX_VOL, VEF_TAKE_LIMIT - pos - buy_used)
            if qty <= 0:
                break
            orders.append(Order(VEF_SYMBOL, ask_px, qty))
            buy_used += qty

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px <= fair_value + 1 or pos - sell_used <= -VEF_TAKE_LIMIT:
                break
            avail = od.buy_orders[bid_px]
            qty = min(avail, VEF_TAKE_MAX_VOL, VEF_TAKE_LIMIT + pos - sell_used)
            if qty <= 0:
                break
            orders.append(Order(VEF_SYMBOL, bid_px, -qty))
            sell_used += qty

        # ── B2) Wide-spread taking at fair value — Change 8 ───────────────────
        if current_spread >= 4:
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

        # ── C) Passive overbid / undercut, 2 layers — Change 7 ────────────────
        passive_bid = best_bid + 1
        passive_ask = best_ask - 1

        # Constraint: bid strictly below fair value, ask strictly above
        max_bid = math.ceil(fair_value) - 1
        min_ask = math.floor(fair_value) + 1
        passive_bid = min(passive_bid, max_bid)
        passive_ask = max(passive_ask, min_ask)

        passive_bid_vol = 0
        passive_ask_vol = 0

        if passive_bid < passive_ask:
            rem_buy  = VEF_POSITION_LIMIT - pos - buy_used
            rem_sell = VEF_POSITION_LIMIT + pos - sell_used

            l1_bid = min(VEF_PASSIVE_VOL, rem_buy)
            l1_ask = min(VEF_PASSIVE_VOL, rem_sell)
            if l1_bid > 0:
                orders.append(Order(VEF_SYMBOL, passive_bid, l1_bid))
                passive_bid_vol += l1_bid
            if l1_ask > 0:
                orders.append(Order(VEF_SYMBOL, passive_ask, -l1_ask))
                passive_ask_vol += l1_ask

            l2_bid_px = min(passive_bid - 2, max_bid)
            l2_ask_px = max(passive_ask + 2, min_ask)
            if l2_bid_px < l2_ask_px:
                l2_bid = min(VEF_PASSIVE_VOL, rem_buy - passive_bid_vol)
                l2_ask = min(VEF_PASSIVE_VOL, rem_sell - passive_ask_vol)
                if l2_bid > 0:
                    orders.append(Order(VEF_SYMBOL, l2_bid_px, l2_bid))
                    passive_bid_vol += l2_bid
                if l2_ask > 0:
                    orders.append(Order(VEF_SYMBOL, l2_ask_px, -l2_ask))
                    passive_ask_vol += l2_ask

        # ── D) Flatten ─────────────────────────────────────────────────────────
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

        # ── E) End-of-day aggressive flatten — Change 4 ───────────────────────
        if state.timestamp > 950000:
            if pos > 0:
                eod_qty = min(pos, VEF_POSITION_LIMIT + pos)
                if eod_qty > 0:
                    orders.append(Order(VEF_SYMBOL, best_bid, -eod_qty))
            elif pos < 0:
                eod_qty = min(abs(pos), VEF_POSITION_LIMIT - pos)
                if eod_qty > 0:
                    orders.append(Order(VEF_SYMBOL, best_ask, eod_qty))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    def _trade_all_vev(
        self, state: TradingState, s: dict
    ) -> Dict[str, List[Order]]:
        """
        Per-strike EMA IV market-making for all VEV options.
        VEV_5400, VEV_5500, VEV_6000, VEV_6500 always return empty.
        VEV_5100 and VEV_5200 get double volume (local_passive_vol = 80).
        """
        result: Dict[str, List[Order]] = {
            sym: []
            for sym in state.order_depths
            if sym.startswith("VEV_")
        }

        vef_od: Optional[OrderDepth] = state.order_depths.get(VEF_SYMBOL)
        if not vef_od or not vef_od.buy_orders or not vef_od.sell_orders:
            return result
        S = (max(vef_od.buy_orders) + min(vef_od.sell_orders)) / 2.0
        T = VEV_T
        timestamp = state.timestamp

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

            # Change 2 + 9: per-symbol local volumes
            if sym in ("VEV_5100", "VEV_5200"):
                local_passive_vol = PASSIVE_VOLUME * 2   # 80
                local_take_vol    = TAKE_VOLUME * 2       # 80
            else:
                local_passive_vol = PASSIVE_VOLUME        # 40
                local_take_vol    = TAKE_VOLUME            # 40

            # ── Fair value and delta ───────────────────────────────────────────
            if midprice <= intrinsic + 0.5:
                if sym in ("VEV_4000", "VEV_4500"):
                    theo_price = S - K
                    delta = 1.0
                else:
                    continue
            else:
                obs_iv = implied_vol(midprice, S, K, T)
                if obs_iv is None or obs_iv < 0.01 or obs_iv > 3.0:
                    continue

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

            pos = state.position.get(sym, 0)
            orders: List[Order] = []

            # ── A) Aggressive taking — Change 6: 0.3% threshold ───────────────
            min_deviation = max(0.3, theo_price * 0.003)

            if best_ask < theo_price - min_deviation:
                qty = min(local_take_vol, INTERNAL_POS_LIMIT - pos)
                if qty > 0:
                    orders.append(Order(sym, best_ask, qty))

            if best_bid > theo_price + min_deviation:
                qty = min(local_take_vol, INTERNAL_POS_LIMIT + pos)
                if qty > 0:
                    orders.append(Order(sym, best_bid, -qty))

            # ── B) Front-of-queue overbid/undercut BEFORE theo quotes — Change 10
            front_ob_vol = 0
            front_uc_vol = 0
            if best_bid + 1 < theo_price:
                fob = min(local_passive_vol, VEV_POS_LIMIT - pos)
                if fob > 0:
                    orders.append(Order(sym, best_bid + 1, fob))
                    front_ob_vol = fob
            if best_ask - 1 > theo_price:
                fuc = min(local_passive_vol, VEV_POS_LIMIT + pos)
                if fuc > 0:
                    orders.append(Order(sym, best_ask - 1, -fuc))
                    front_uc_vol = fuc

            # ── C) Theo-based passive quotes (2 layers) ───────────────────────
            bid_px = int(math.floor(theo_price)) - 1
            ask_px = int(math.ceil(theo_price)) + 1

            bid_vol = 0
            ask_vol = 0
            l2_bid  = 0
            l2_ask  = 0

            if bid_px < ask_px:
                delta_multiplier = max(0.5, min(2.0, delta * 2))
                skew = int((pos / INTERNAL_POS_LIMIT) * 2 * delta_multiplier)
                bid_px -= skew
                ask_px -= skew

                bid_vol = min(local_passive_vol, VEV_POS_LIMIT - pos - front_ob_vol)
                ask_vol = min(local_passive_vol, VEV_POS_LIMIT + pos - front_uc_vol)

                if bid_vol > 0 and bid_px < theo_price:
                    orders.append(Order(sym, bid_px, bid_vol))
                if ask_vol > 0 and ask_px > theo_price:
                    orders.append(Order(sym, ask_px, -ask_vol))

                second_bid = bid_px - 2
                second_ask = ask_px + 2
                rem_buy  = VEV_POS_LIMIT - pos - front_ob_vol - bid_vol
                rem_sell = VEV_POS_LIMIT + pos - front_uc_vol - ask_vol
                l2_bid = min(local_passive_vol, rem_buy)
                l2_ask = min(local_passive_vol, rem_sell)
                if l2_bid > 0 and second_bid < theo_price:
                    orders.append(Order(sym, second_bid, l2_bid))
                if l2_ask > 0 and second_ask > theo_price:
                    orders.append(Order(sym, second_ask, -l2_ask))

            # ── D) Residual overbid/undercut capacity ─────────────────────────
            ob_cap = VEV_POS_LIMIT - pos - front_ob_vol - bid_vol - l2_bid
            uc_cap = VEV_POS_LIMIT + pos - front_uc_vol - ask_vol - l2_ask
            if best_bid + 1 < theo_price and ob_cap > 0:
                orders.append(Order(sym, best_bid + 1, min(20, ob_cap)))
            if best_ask - 1 > theo_price and uc_cap > 0:
                orders.append(Order(sym, best_ask - 1, -min(20, uc_cap)))

            # ── E) Flatten ─────────────────────────────────────────────────────
            if abs(pos) > FLATTEN_THRESHOLD_VEV:
                flatten_px = int(round(theo_price))
                flatten_qty = min(10, abs(pos) - FLATTEN_TARGET_VEV)
                if flatten_qty > 0:
                    if pos > 0:
                        orders.append(Order(sym, flatten_px, -flatten_qty))
                    else:
                        orders.append(Order(sym, flatten_px, flatten_qty))

            # ── F) End-of-day aggressive flatten — Change 5 ───────────────────
            if timestamp > 950000 and abs(pos) > 0:
                if pos > 0:
                    orders.append(Order(sym, best_bid, -min(pos, 50)))
                else:
                    orders.append(Order(sym, best_ask, min(abs(pos), 50)))

            result[sym] = orders

        return result
