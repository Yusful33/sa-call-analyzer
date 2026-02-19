"""Travel agent use-case: shared prompts, queries, guardrails, and tools."""

QUERIES = [
    "I need a weekend trip to Paris for two in March, mid-range budget. Flights and a central hotel.",
    "Find me direct flights from NYC to London next month and a 4-star hotel near the business district.",
    "Plan a 5-day vacation to Tokyo: flights, hotel with good transit access, and one day-tour suggestion.",
    "Best way to get from San Francisco to Barcelona in June? Prefer non-stop or one short layover.",
    "I want a beach holiday in Greece for 7 days — flights and a family-friendly hotel with a pool.",
]

SYSTEM_PROMPT = (
    "You are a knowledgeable travel agent. Use the available tools to look up flights and hotels, "
    "then provide clear, accurate recommendations with dates, prices, and practical details. "
    "Always summarize options and mention any important restrictions or tips."
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


def flight_search(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date."""
    import json as _json
    return _json.dumps({
        "origin": origin,
        "destination": destination,
        "date": date,
        "options": [
            {"airline": "Air Example", "departure": "08:00", "arrival": "11:30", "price_usd": 289, "stops": 0},
            {"airline": "Budget Fly", "departure": "14:20", "arrival": "18:45", "price_usd": 199, "stops": 1},
        ],
        "currency": "USD",
    })


def hotel_search(city: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Search for hotels in a city for given check-in/out dates and guest count."""
    import json as _json
    return _json.dumps({
        "city": city,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "options": [
            {"name": "Central Plaza Hotel", "stars": 4, "price_per_night_usd": 145, "rating": 4.2},
            {"name": "City View Inn", "stars": 3, "price_per_night_usd": 89, "rating": 4.0},
        ],
        "currency": "USD",
    })


EVALUATORS = [
    {
        "name": "travel_recommendation_quality",
        "criteria": "accuracy and usefulness of travel options (correct cities, dates, and realistic prices)",
    },
]
