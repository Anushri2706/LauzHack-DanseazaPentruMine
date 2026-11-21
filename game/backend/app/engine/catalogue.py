"""Game catalogue: habits, one-off items, homes, rental flats, joy table and housing rules.

Every number here becomes an editable assumption (🟡 "ours") in assumptions.META, so the
Settings app can show and change it. The engine never reads these constants directly; it reads
the params dict built from them, so an edited price flows through every calculation.

Prices are in month-0 CHF. Habit costs are inflated like other living costs; one-off item
prices are paid as listed (month-0 price × inflation at the month of purchase).
Purchases are CONSUMPTION, not assets: they reduce cash and net worth.

Joy points are a game preference score: "not money, not a measure of a good life".
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HabitLevel:
    label: str
    cost: float      # CHF/month, on top of base living costs (month-0 prices)
    joy: float       # joy points per month


@dataclass(frozen=True)
class Habit:
    id: str
    label: str
    place: str
    levels: tuple[HabitLevel, ...]


@dataclass(frozen=True)
class Item:
    id: str
    label: str
    place: str
    price: float     # CHF one-off (month-0 prices)
    joy: float       # joy points, once
    kind: str        # "purchase" | "trip"
    abroad_only: bool = False


@dataclass(frozen=True)
class Home:
    id: str
    label: str
    price: float     # CHF, Lausanne prices; scaled by canton for other personas


# Level 0 of every habit means "nothing on top of the base living costs from the profile/starter kit",
# so a player who never touches habits gets exactly the old engine (and the starter-kit cross-check).
HABITS: dict[str, Habit] = {h.id: h for h in (
    Habit("eat_out", "Eating out", "cafe", (
        HabitLevel("Never", 0, 0), HabitLevel("1× per week", 170, 1),
        HabitLevel("2× per week", 340, 2), HabitLevel("4× per week", 680, 3))),
    Habit("shopping", "Shopping budget", "shop", (
        HabitLevel("Basics only (in living costs)", 0, 0), HabitLevel("Low", 80, 0.5),
        HabitLevel("Medium", 200, 1), HabitLevel("High", 450, 1.5))),
    Habit("gym", "Gym membership", "shop", (
        HabitLevel("No gym", 0, 0), HabitLevel("Member", 60, 1))),
)}

ITEMS: dict[str, Item] = {i.id: i for i in (
    Item("bike", "Bike", "shop", 800, 6, "purchase"),
    Item("laptop", "Laptop", "shop", 1500, 4, "purchase"),
    Item("phone", "Phone", "shop", 900, 3, "purchase"),
    Item("clothes", "New clothes", "shop", 200, 2, "purchase"),
    Item("furniture", "Furniture", "shop", 2000, 5, "purchase"),
    Item("treat_friends", "Treat your friends to dinner", "cafe", 120, 3, "purchase"),
    Item("weekend_trip", "Weekend trip", "station", 450, 6, "trip"),
    Item("flight_home", "Flight home to see family", "station", 250, 5, "trip", abroad_only=True),
)}

HOMES: dict[str, Home] = {h.id: h for h in (
    Home("studio", "Studio", 450_000),
    Home("flat", "3-room flat", 750_000),
    Home("house", "Small house", 1_100_000),
)}

# Change flat: rent factor applied to the base rent of where you live (Switzerland or abroad).
FLATS: dict[str, tuple[str, float]] = {
    "cheaper": ("Cheaper flat", 0.8),
    "same": ("Similar flat", 1.0),
    "nicer": ("Nicer flat", 1.3),
}

# Rough property price level by canton relative to Lausanne/VD. Our assumption, not statistics.
CANTON_PRICE_LEVEL = {"ZH": 1.25, "GE": 1.2, "VD": 1.0, "BS": 0.95, "LU": 0.95, "BE": 0.8,
                      "TI": 0.8, "FR": 0.75, "SG": 0.75, "NE": 0.65}

# key: (default, label, unit, group, min, max, note)
RULES: dict[str, tuple] = {
    # Housing (simple mortgage)
    "mortgage_rate": (0.02, "Mortgage interest", "%/year", "Housing", 0, 0.15, ""),
    "down_payment_share": (0.20, "Down payment", "% of price", "Housing", 0.05, 1, ""),
    "purchase_fee_rate": (0.02, "Purchase fees (notary, tax)", "% of price", "Housing", 0, 0.1, ""),
    "sale_fee_rate": (0.03, "Selling fees", "% of home value", "Housing", 0, 0.1, ""),
    "amortization_annual": (0.01, "Mortgage amortization", "% of price per year", "Housing", 0, 0.1,
                            "A transfer: reduces cash and mortgage equally, net worth unchanged."),
    "maintenance_annual": (0.01, "Home maintenance", "% of price per year", "Housing", 0, 0.05, ""),
    "home_growth_annual": (0.015, "Home value growth", "%/year", "Housing", -0.1, 0.1, ""),
    "flat_moving_cost": (1500, "Moving cost when changing flat", "CHF", "Housing", 0, None, ""),
    # Work
    "raise_ask_size": (0.05, "Raise if you ask and it works", "% of salary", "Work", 0, 0.5, ""),
    "raise_ask_chance": (0.5, "Chance that asking for a raise works", "%", "Work", 0, 1,
                         "Drawn once from your seeded world, so it is reproducible."),
    "raise_ask_min_months": (24, "Months in the job before asking for a raise", "months", "Work", 0, 120, ""),
    "reduced_hours_share": (0.8, "Reduced hours", "% of full time", "Work", 0.2, 1, ""),
    "quit_benefit_rate": (0.0, "Income after quitting your job", "% of salary", "Work", 0, 1,
                          "We assume no benefits right after quitting voluntarily."),
    # Joy (a game preference score, not money)
    "joy_enabled": (1, "Count joy points (0 = off)", "on/off", "Joy", 0, 1,
                    "Joy is a game preference score: not money, not a measure of a good life."),
    "joy_studying": (1, "Joy while studying", "points/month", "Joy", -10, 10, ""),
    "joy_working": (0, "Joy while working", "points/month", "Joy", -10, 10, ""),
    "joy_abroad": (2, "Joy while living abroad", "points/month", "Joy", -10, 10, ""),
    "joy_unemployed": (-2, "Joy while between jobs", "points/month", "Joy", -10, 10, ""),
    "joy_retired": (1, "Joy while retired", "points/month", "Joy", -10, 10, ""),
    "joy_home_owner": (1, "Joy from owning your home", "points/month", "Joy", -10, 10, ""),
    "joy_reduced_hours": (1, "Joy from working reduced hours", "points/month", "Joy", -10, 10, ""),
}


def habit_cost_key(hid: str, level: int) -> str:
    return f"habit_{hid}_{level}_cost"


def habit_joy_key(hid: str, level: int) -> str:
    return f"habit_{hid}_{level}_joy"


def item_price_key(iid: str) -> str:
    return f"item_{iid}_price"


def item_joy_key(iid: str) -> str:
    return f"item_{iid}_joy"


def home_price_key(hid: str) -> str:
    return f"home_{hid}_price"


def flat_factor_key(fid: str) -> str:
    return f"flat_{fid}_factor"


def catalogue_assumptions() -> list[tuple]:
    """(key, default, label, unit, group, min, max, note) for every catalogue number."""
    out = []
    for h in HABITS.values():
        for i, lv in enumerate(h.levels):
            if i == 0:
                continue  # level 0 is "nothing extra": fixed at 0, not an assumption
            out.append((habit_cost_key(h.id, i), lv.cost, f"{h.label}: {lv.label}", "CHF/month",
                        "Habits", 0, None, ""))
            out.append((habit_joy_key(h.id, i), lv.joy, f"{h.label}: {lv.label} (joy)", "points/month",
                        "Joy", -10, 10, ""))
    for it in ITEMS.values():
        out.append((item_price_key(it.id), it.price, it.label, "CHF one-off", "Purchases & trips", 0, None,
                    "Consumption, not an asset."))
        out.append((item_joy_key(it.id), it.joy, f"{it.label} (joy)", "points once", "Joy", -50, 50, ""))
    for hm in HOMES.values():
        out.append((home_price_key(hm.id), hm.price, f"Home price: {hm.label}", "CHF", "Housing", 0, None,
                    "Lausanne level; other cantons scaled by a rough price level."))
    for fid, (label, factor) in FLATS.items():
        out.append((flat_factor_key(fid), factor, f"Rent factor: {label}", "× base rent", "Housing", 0.3, 3, ""))
    for key, (default, label, unit, group, lo, hi, note) in RULES.items():
        out.append((key, default, label, unit, group, lo, hi, note))
    return out
