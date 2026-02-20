"""Travel agent use-case: shared prompts, queries, guardrails, and tools."""

import hashlib
import json
import random
import re
from datetime import datetime, timedelta

# Queries that trigger only flight_search (no hotel wording). Good for evaluators to see single-tool traces.
QUERIES_FLIGHT_ONLY = [
    "Best way to get from San Francisco to Barcelona in June? Prefer non-stop or one short layover.",
    "How do I get from LA to Chicago next week? Direct flights only.",
    "Cheapest flights from Boston to Miami in August? Willing to do one stop.",
    "Find me direct flights from NYC to London next month.",
]

# Queries that trigger only hotel_search (accommodation only). Good for evaluator mix.
QUERIES_HOTEL_ONLY = [
    "Where should I stay in Paris for a romantic weekend? Prefer something central.",
    "Need a hotel in London with a gym and breakfast for 3 nights.",
    "Recommend a family-friendly place to stay in Athens with a pool.",
]

# Queries that trigger both flight_search and hotel_search.
QUERIES_BOTH = [
    "I need a weekend trip to Paris for two in March, mid-range budget. Flights and a central hotel.",
    "Find me direct flights from NYC to London next month and a 4-star hotel near the business district.",
    "Plan a 5-day vacation to Tokyo: flights, hotel with good transit access, and one day-tour suggestion.",
    "I want a beach holiday in Greece for 7 days — flights and a family-friendly hotel with a pool.",
]

# Union of all (backward compat); order doesn't matter for random.choice.
QUERIES = list(QUERIES_FLIGHT_ONLY + QUERIES_HOTEL_ONLY + QUERIES_BOTH)


def sample_travel_query(
    *,
    flight_only_weight: float = 0.35,
    hotel_only_weight: float = 0.25,
    both_weight: float = 0.40,
) -> str:
    """
    Sample a query so traces include a mix of single-tool and both-tool calls.
    Use these weights so evaluators see both patterns (e.g. "only call hotel when user asks for accommodation").
    """
    r = random.random()
    if r < flight_only_weight and QUERIES_FLIGHT_ONLY:
        return random.choice(QUERIES_FLIGHT_ONLY)
    if r < flight_only_weight + hotel_only_weight and QUERIES_HOTEL_ONLY:
        return random.choice(QUERIES_HOTEL_ONLY)
    return random.choice(QUERIES_BOTH) if QUERIES_BOTH else random.choice(QUERIES)

# Defaults when query doesn't specify (we use flexible search and state assumptions)
DEFAULT_ORIGIN = "NYC"
DEFAULT_DESTINATION = "Paris"
DEFAULT_DATE = "2025-03-15"
DEFAULT_CHECK_IN = "2025-03-15"
DEFAULT_CHECK_OUT = "2025-03-17"
DEFAULT_GUESTS = 2

# City name variations for common destinations (query may say "Paris" or "Barcelona")
CITY_ALIASES = {
    "paris": "Paris",
    "london": "London",
    "tokyo": "Tokyo",
    "barcelona": "Barcelona",
    "greece": "Athens",
    "athens": "Athens",
    "nyc": "NYC",
    "new york": "NYC",
    "san francisco": "SFO",
    "sf": "SFO",
    "sfo": "SFO",
}

# Month name -> first day of month for that year (demo)
MONTH_TO_DATE = {
    "march": "2025-03-15",
    "april": "2025-04-01",
    "may": "2025-05-01",
    "june": "2025-06-15",
    "july": "2025-07-01",
    "august": "2025-08-01",
    "next month": "2025-04-01",
}


def select_tools_for_query(query: str) -> tuple[bool, bool]:
    """
    Decide whether to run flight_search and/or hotel_search based on the user query.
    Returns (run_flight_search, run_hotel_search).
    """
    q = (query or "").lower()
    # Explicit hotel/accommodation intent
    hotel_keywords = ("hotel", "stay", "accommodation", "lodging", "where to stay", "place to stay", "room")
    want_hotel = any(k in q for k in hotel_keywords)
    # Explicit flight/transport intent
    flight_keywords = ("flight", "fly", "direct flight", "get from", "way to get", "how to get", "non-stop", "layover")
    want_flight = any(k in q for k in flight_keywords)
    # Trip/vacation phrasing often implies both
    trip_both = any(x in q for x in ("trip", "vacation", "holiday", "weekend trip", "flights and", "flight and a"))
    if want_hotel and not want_flight and not trip_both:
        return (False, True)
    if want_flight and not want_hotel and not trip_both:
        return (True, False)
    # Default: both (e.g. "flights and hotel", or ambiguous)
    return (True, True)


def parse_travel_query(query: str) -> dict:
    """
    Extract origin, destination, dates, and guests from the user query.
    Uses heuristics; when under-specified returns defaults and caller should state assumptions.
    """
    q = (query or "").lower().strip()
    origin = DEFAULT_ORIGIN
    destination = DEFAULT_DESTINATION
    date_out = DEFAULT_DATE
    check_in = DEFAULT_CHECK_IN
    check_out = DEFAULT_CHECK_OUT
    guests = DEFAULT_GUESTS
    assumptions = []

    # "from X to Y" or "X to Y" or "to Barcelona" or "holiday in Greece" / "trip to Tokyo"
    from_to = re.search(r"from\s+([a-z\s]+?)\s+to\s+([a-z\s]+?)(?:\s|,|\.|$)", q, re.I)
    if from_to:
        origin = from_to.group(1).strip().upper()
        destination = from_to.group(2).strip().title()
    else:
        to_match = re.search(r"\bto\s+([a-z\s]+?)(?:\s|,|\.|for|in|$)", q, re.I)
        if to_match:
            destination = to_match.group(1).strip().title()
        else:
            # "in Greece", "holiday in Greece", "vacation in Barcelona"
            in_place = re.search(r"\b(?:holiday|vacation|trip|week|stay|beach)\s+in\s+([a-z]+)", q, re.I)
            if not in_place:
                in_place = re.search(r"\bin\s+([a-z]+?)(?:\s+for\s|\s*—|\s*,|\.|$)", q, re.I)
            if in_place:
                destination = in_place.group(1).strip().title()

    # Normalize origin/destination to tool-friendly codes/names
    for alias, city in CITY_ALIASES.items():
        if alias in origin.lower():
            origin = city
            break
    for alias, city in CITY_ALIASES.items():
        if alias in destination.lower():
            destination = city
            break

    # Dates: "in March", "in June", "next month"
    for month_key, date_str in MONTH_TO_DATE.items():
        if month_key in q:
            date_out = date_str
            check_in = date_str
            # check_out = same month + 2–3 days for weekend, or +7 for week
            if "week" in q or "5-day" in q or "5 day" in q or "7 days" in q:
                try:
                    d = datetime.strptime(date_str, "%Y-%m-%d")
                    if "7" in q:
                        end = d + timedelta(days=7)
                        check_out = end.strftime("%Y-%m-%d")
                    else:
                        end = d + timedelta(days=3)
                        check_out = end.strftime("%Y-%m-%d")
                except Exception:
                    check_out = date_str[:8] + "18"
            else:
                check_out = date_str[:8] + "17"
            break

    # Guests: "for two", "2 guests", "two people"
    if "for two" in q or " two " in q or "2 guests" in q or "two people" in q:
        guests = 2
    elif " for 3 " in q or "3 guests" in q:
        guests = 3

    if origin == DEFAULT_ORIGIN and "nyc" not in q and "new york" not in q:
        assumptions.append("origin (assuming NYC)")
    if destination == DEFAULT_DESTINATION and "paris" not in q:
        assumptions.append("destination (assuming Paris)")
    if date_out == DEFAULT_DATE and "march" not in q:
        assumptions.append("dates (assuming March 2025 weekend)")

    # Hotel search city: destination of the trip (not origin)
    city = destination if destination not in ("NYC", "SFO") else "Paris"

    return {
        "origin": origin,
        "destination": destination,
        "city": city,
        "date_out": date_out,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "assumptions": assumptions,
    }


def build_options_table(flight_json_str: str, hotel_json_str: str) -> str:
    """
    Parse flight and hotel JSON and return a short comparison table (and raw options)
    so the LLM can summarize them instead of ignoring tool results.
    """
    lines = []
    try:
        flight = json.loads(flight_json_str)
        options = flight.get("options", [])
        if options:
            lines.append("**Flights**")
            lines.append("| Airline | Departure | Arrival | Price (USD) | Stops |")
            lines.append("|---------|-----------|---------|--------------|-------|")
            for o in options:
                lines.append(
                    f"| {o.get('airline', '')} | {o.get('departure', '')} | {o.get('arrival', '')} | "
                    f"{o.get('price_usd', '')} | {o.get('stops', 0)} |"
                )
            lines.append("")
    except (json.JSONDecodeError, TypeError):
        lines.append("(Flight results could not be parsed.)\n")

    try:
        hotel = json.loads(hotel_json_str)
        options = hotel.get("options", [])
        if options:
            lines.append("**Hotels**")
            lines.append("| Name | Stars | Price/night (USD) | Rating |")
            lines.append("|------|-------|-------------------|--------|")
            for o in options:
                lines.append(
                    f"| {o.get('name', '')} | {o.get('stars', '')} | "
                    f"{o.get('price_per_night_usd', '')} | {o.get('rating', '')} |"
                )
            lines.append("")
    except (json.JSONDecodeError, TypeError):
        lines.append("(Hotel results could not be parsed.)\n")

    return "\n".join(lines) if lines else "(No options to display.)"


SYSTEM_PROMPT = (
    "You are a knowledgeable travel agent. You receive tool results (flights and hotels) as structured tables. "
    "Your job is to summarize these options clearly for the user: compare prices, mention best value or best for "
    "convenience, and give a short recommendation. If the system had to assume dates or cities (assumptions are "
    "listed), state them briefly (e.g. 'I assumed March 2025 and NYC→Paris.'). Do NOT ask the user for basic "
    "info they already provided or that was assumed—give them the comparison and a concrete recommendation."
)

GUARDRAILS = [
    {
        "name": "travel_safety_check",
        "system_prompt": (
            "You are a travel request validator. Check if the user input requests only "
            "legal, safe travel options (no prohibited destinations or unsafe itineraries). "
            "Respond ONLY 'PASS' or 'FAIL: <reason>'."
        ),
    },
]


# Mock data seeds by route/city for deterministic but varied, parameter-aware results.
_FLIGHT_TEMPLATES = [
    {"airline": "SkyDirect", "departure": "06:30", "arrival": "09:45", "stops": 0, "base_price": 320},
    {"airline": "BudgetFly", "departure": "12:15", "arrival": "16:20", "stops": 1, "base_price": 189},
    {"airline": "GlobalAir", "departure": "18:00", "arrival": "21:10", "stops": 0, "base_price": 355},
    {"airline": "EuroWings", "departure": "09:45", "arrival": "14:30", "stops": 1, "base_price": 215},
]
_HOTEL_BY_CITY = {
    "athens": [
        {"name": "Acropolis View Hotel", "stars": 4, "base_price": 165, "rating": 4.5, "tags": ["pool", "family-friendly"]},
        {"name": "Plaka Suites", "stars": 4, "base_price": 142, "rating": 4.3, "tags": ["pool", "family-friendly"]},
        {"name": "Aegean Breeze Resort", "stars": 5, "base_price": 285, "rating": 4.7, "tags": ["pool", "beach", "family-friendly"]},
        {"name": "City Center Athens", "stars": 3, "base_price": 89, "rating": 4.0, "tags": []},
    ],
    "paris": [
        {"name": "Central Plaza Paris", "stars": 4, "base_price": 195, "rating": 4.4, "tags": []},
        {"name": "City View Inn Paris", "stars": 3, "base_price": 112, "rating": 4.1, "tags": []},
        {"name": "Le Marais Hotel", "stars": 4, "base_price": 178, "rating": 4.3, "tags": []},
    ],
    "london": [
        {"name": "Thames Riverside Hotel", "stars": 4, "base_price": 188, "rating": 4.3, "tags": []},
        {"name": "West End Suites", "stars": 3, "base_price": 125, "rating": 4.0, "tags": []},
    ],
    "tokyo": [
        {"name": "Shibuya Central Hotel", "stars": 4, "base_price": 165, "rating": 4.4, "tags": []},
        {"name": "Tokyo Transit Inn", "stars": 3, "base_price": 98, "rating": 4.1, "tags": []},
    ],
    "barcelona": [
        {"name": "Ramblas Hotel", "stars": 4, "base_price": 155, "rating": 4.4, "tags": []},
        {"name": "Gothic Quarter Suites", "stars": 3, "base_price": 102, "rating": 4.0, "tags": []},
    ],
}


def _route_multiplier(origin: str, destination: str) -> float:
    """Return a price multiplier so long-haul routes cost more (deterministic from route)."""
    key = f"{origin.lower()}|{destination.lower()}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return 0.85 + (h % 30) / 100.0


def _date_season_multiplier(date_str: str) -> float:
    """Slightly higher prices in summer months (deterministic)."""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        if d.month in (6, 7, 8):
            return 1.15
        if d.month in (12, 1):
            return 1.08
    except Exception:
        pass
    return 1.0


def flight_search(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date.
    Returns parameter-aware mock results: route, date, and number of options vary with inputs.
    """
    origin = (origin or "").strip()
    destination = (destination or "").strip()
    date = (date or "")[:10]
    mult = _route_multiplier(origin, destination) * _date_season_multiplier(date)
    seed = hash(f"{origin}|{destination}|{date}") % (2**31)
    # Pick 2–4 options deterministically
    n = 2 + (seed % 3)
    options = []
    for i in range(n):
        t = _FLIGHT_TEMPLATES[(seed + i) % len(_FLIGHT_TEMPLATES)]
        price = max(99, int(t["base_price"] * mult) + (i * 17) % 50)
        options.append({
            "airline": t["airline"],
            "departure": t["departure"],
            "arrival": t["arrival"],
            "price_usd": price,
            "stops": t["stops"],
        })
    options.sort(key=lambda o: o["price_usd"])
    return json.dumps({
        "origin": origin or "Unknown",
        "destination": destination or "Unknown",
        "date": date or "Unknown",
        "options": options,
        "currency": "USD",
    })


def hotel_search(city: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Search for hotels in a city for given check-in/out dates and guest count.
    Returns parameter-aware mock results: city-specific names, prices vary by city/dates/guests.
    """
    city = (city or "").strip()
    check_in = (check_in or "")[:10]
    check_out = (check_out or "")[:10]
    guests = max(1, int(guests) if isinstance(guests, (int, float)) else 1)
    key = city.lower()
    for alias, canonical in CITY_ALIASES.items():
        if alias in key or key in alias:
            key = canonical.lower()
            break
    else:
        key = key or "paris"
    hotels = _HOTEL_BY_CITY.get(key, _HOTEL_BY_CITY["paris"])
    mult = _date_season_multiplier(check_in)
    if guests > 2:
        mult *= 1.05
    seed = hash(f"{city}|{check_in}|{check_out}|{guests}") % (2**31)
    nights = 1
    try:
        if check_in and check_out:
            d1 = datetime.strptime(check_in[:10], "%Y-%m-%d")
            d2 = datetime.strptime(check_out[:10], "%Y-%m-%d")
            nights = max(1, (d2 - d1).days)
    except Exception:
        pass
    options = []
    for i, h in enumerate(hotels[:4]):
        price = max(60, int(h["base_price"] * mult) + (seed + i) % 25)
        opt = {
            "name": h["name"],
            "stars": h["stars"],
            "price_per_night_usd": price,
            "rating": h["rating"],
        }
        if h.get("tags"):
            opt["amenities"] = ", ".join(h["tags"])
        options.append(opt)
    return json.dumps({
        "city": city or "Unknown",
        "check_in": check_in or "Unknown",
        "check_out": check_out or "Unknown",
        "guests": guests,
        "nights": nights,
        "options": options,
        "currency": "USD",
    })


EVALUATORS = [
    {
        "name": "travel_recommendation_quality",
        "criteria": "accuracy and usefulness of travel options (correct cities, dates, and realistic prices)",
    },
]
