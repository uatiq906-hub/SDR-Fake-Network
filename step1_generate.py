"""
step1_generate.py — the complete generator. One file, nothing to edit.

Everything is built in:
    58 situations
    14 unit pairs
     9 event thread templates
     6 rotating example styles
     detail slots for time, weather, signal, urgency, texture, operator

    58 x 14 = 812 base combinations, multiplied again by the detail slots.

Features:
    - Balanced sampling, so coverage is even rather than clumped
    - Live deduplication, catching repetition as it happens
    - Avoid-list, telling the model which openings it has overused
    - Optional quality scoring by a second model
    - Event threads: linked exchanges that develop over time
    - Resumable. Ctrl+C is safe, re-running continues where it stopped
    - Progress with ETA

Requires:
    pip install requests

Run:
    python step1_generate.py 500              generate 500 exchanges
    python step1_generate.py 500 --fresh      ignore existing file, start over
    python step1_generate.py 150 --threads    generate event threads instead
    python step1_generate.py 500 --judge      enable quality scoring (slower)

Recommended sequence for a full pool:

    python step1_generate.py 350 --fresh
    python step1_generate.py 500 --threads

Output:
    exchanges.json
"""

import json
import os
import random
import re
import signal
import sys
import time
from collections import Counter

import requests


# ===========================================================================
# CONFIGURATION
# ===========================================================================

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

GEN_MODEL = "llama3.1:8b"      # writes the traffic
JUDGE_MODEL = "llama3.1:8b"    # scores it, only used with --judge

OUTPUT_FILE = "exchanges.json"

# --- The settings you will actually change -------------------------------

HOW_MANY = 500          # how many good exchanges you want in total
MAX_ATTEMPTS = 1500     # give up after this many tries, so it cannot run forever

MODE = "single"         # "single" for standalone exchanges, "threads" for
                        # linked event sequences that develop over time

FRESH_START = False     # True  = ignore exchanges.json and start over
                        # False = continue adding to whatever is already there

ENABLE_JUDGE = False    # True = score each exchange with JUDGE_MODEL.
                        # Roughly doubles the runtime. Worth it once you
                        # have a larger model to judge with.

# --- Settings you will rarely change --------------------------------------

USE_JSON = True                # set False if writing quality disappoints
MIN_SCORE = 4                  # judge scores 1-5; keep 4 and above
MAX_CONSECUTIVE_FAILS = 40     # stop early if something is clearly broken

# Command line still works and overrides the above, for convenience:
#   python step1_generate.py 250 --fresh --threads --judge


# ===========================================================================
# SITUATIONS — 58 of them
#
# THE TEST FOR ADDING ONE: would the conversation have a different SHAPE?
#
#   Same shape:      "spotted a truck" / "spotted a car" / "spotted a van"
#   Different shape: "requesting resupply" / "reporting a breakdown"
#
# If two entries produce structurally identical traffic with one noun
# swapped, they are one situation, and the model writes nearly the same
# text for both.
#
# Note how many are dull. That is deliberate. A real net is roughly 90
# percent administration and 10 percent incident. A pool made mostly of
# contact reports sounds fake within a minute of listening.
# ===========================================================================

SITUATIONS = [

    # --- Communications procedure itself (the most common traffic of all) ---
    {"id": "radio_check", "text": "{caller} carries out a routine communications check with {other} at the start of the shift.", "starter": "caller", "min": 2, "max": 4},
    {"id": "signal_strength", "text": "{caller} asks {other} for a signal strength and readability report.", "starter": "caller", "min": 2, "max": 4},
    {"id": "poor_signal", "text": "{other} is receiving {caller} badly and asks for the message to be sent again.", "starter": "caller", "min": 4, "max": 7},
    {"id": "move_for_comms", "text": "{caller} is told their signal is poor and agrees to move to higher ground and call back.", "starter": "other", "min": 3, "max": 6},
    {"id": "relay_message", "text": "{caller} asks {other} to relay a message to a station they cannot reach directly.", "starter": "caller", "min": 4, "max": 7},
    {"id": "correction", "text": "{caller} realises an earlier report contained a wrong grid reference and corrects it to {grid}.", "starter": "caller", "min": 4, "max": 6},
    {"id": "authentication", "text": "{other} challenges {caller} to authenticate before accepting the message.", "starter": "other", "min": 3, "max": 6},
    {"id": "net_closedown", "text": "{other} announces the net is closing down for the night and confirms the station heard it.", "starter": "other", "min": 3, "max": 5},
    {"id": "wait_out", "text": "{caller} passes a request, is told to wait, and is called back a moment later with an answer.", "starter": "caller", "min": 4, "max": 6},
    {"id": "wrong_station", "text": "{caller} calls the wrong station by mistake and corrects themselves.", "starter": "caller", "min": 3, "max": 5},

    # --- Routine reporting ---
    {"id": "position_report", "text": "{caller} reports their current position as part of routine reporting.", "starter": "caller", "min": 3, "max": 5},
    {"id": "nothing_to_report", "text": "{caller} submits a routine situation report with nothing significant to report.", "starter": "caller", "min": 3, "max": 5},
    {"id": "checkpoint_reached", "text": "{caller} reports reaching their designated checkpoint at grid {grid}.", "starter": "caller", "min": 3, "max": 5},
    {"id": "hourly_report", "text": "{other} calls {caller} for their hourly report and {caller} responds.", "starter": "other", "min": 3, "max": 5},
    {"id": "return_to_base", "text": "{caller} reports the patrol is complete and they are returning to base.", "starter": "caller", "min": 3, "max": 5},
    {"id": "departure_report", "text": "{caller} reports departing their current location and gives an estimated arrival time.", "starter": "caller", "min": 3, "max": 5},
    {"id": "arrival_report", "text": "{caller} reports arriving at their destination and going firm.", "starter": "caller", "min": 2, "max": 4},
    {"id": "shift_handover", "text": "{caller} hands over responsibility for the sector to {other} at the end of their shift.", "starter": "caller", "min": 4, "max": 6},
    {"id": "task_complete", "text": "{caller} reports that the task they were given has been completed.", "starter": "caller", "min": 3, "max": 5},
    {"id": "status_update", "text": "{other} asks {caller} for a general status update and receives a brief one.", "starter": "other", "min": 3, "max": 5},

    # --- Movement and routes ---
    {"id": "movement_permission", "text": "{caller} requests permission to move to a new position at grid {grid}.", "starter": "caller", "min": 4, "max": 6},
    {"id": "route_confirm", "text": "{caller} confirms which route they intend to take back and asks for approval.", "starter": "caller", "min": 3, "max": 6},
    {"id": "obstacle_report", "text": "{caller} reports a blocked route at grid {grid} and suggests an alternative.", "starter": "caller", "min": 4, "max": 6},
    {"id": "route_change", "text": "{other} directs {caller} to change route and {caller} reads the new route back.", "starter": "other", "min": 4, "max": 6},
    {"id": "extend_patrol", "text": "{caller} requests permission to extend their patrol time by two hours.", "starter": "caller", "min": 3, "max": 5},
    {"id": "hold_position", "text": "{other} instructs {caller} to hold their current position until further notice.", "starter": "other", "min": 3, "max": 5},
    {"id": "rendezvous", "text": "{caller} and {other} agree a place and time to meet.", "starter": "caller", "min": 4, "max": 6},
    {"id": "slow_progress", "text": "{caller} reports they are running behind schedule and gives a revised timing.", "starter": "caller", "min": 3, "max": 5},
    {"id": "lost_bearings", "text": "{caller} is unsure of their exact position and asks for help confirming it.", "starter": "caller", "min": 4, "max": 7},

    # --- Logistics and supply ---
    {"id": "resupply_request", "text": "{caller} is running low on water and batteries and requests resupply at their next checkpoint.", "starter": "caller", "min": 4, "max": 7},
    {"id": "fuel_state", "text": "{other} asks {caller} for their vehicle fuel state and {caller} reports it.", "starter": "other", "min": 3, "max": 5},
    {"id": "resupply_confirm", "text": "{caller} confirms a resupply delivery has arrived and been checked.", "starter": "caller", "min": 3, "max": 4},
    {"id": "wrong_supplies", "text": "{caller} reports the wrong items were delivered and asks for the correct ones.", "starter": "caller", "min": 4, "max": 6},
    {"id": "ration_state", "text": "{caller} reports how many days of rations the patrol has remaining.", "starter": "caller", "min": 3, "max": 5},
    {"id": "collection_point", "text": "{other} passes {caller} the location of a supply collection point at grid {grid}.", "starter": "other", "min": 3, "max": 6},

    # --- Equipment and vehicles ---
    {"id": "vehicle_fault", "text": "{caller} has a vehicle with a mechanical fault and needs recovery assistance.", "starter": "caller", "min": 4, "max": 7},
    {"id": "equipment_check", "text": "{caller} reports a faulty piece of equipment and asks for a replacement.", "starter": "caller", "min": 3, "max": 6},
    {"id": "battery_low", "text": "{caller} reports their radio battery is running low and gives an estimate of remaining time.", "starter": "caller", "min": 3, "max": 5},
    {"id": "generator_fault", "text": "{caller} reports a power problem at their location affecting equipment.", "starter": "caller", "min": 3, "max": 6},
    {"id": "recovery_arrived", "text": "{caller} confirms a recovery team has reached them and work has started.", "starter": "caller", "min": 3, "max": 5},

    # --- Personnel and administration ---
    {"id": "headcount", "text": "{other} requests a personnel headcount and {caller} provides it.", "starter": "other", "min": 3, "max": 5},
    {"id": "sick_soldier", "text": "{caller} reports a member of the patrol is unwell and asks for advice.", "starter": "caller", "min": 4, "max": 7},
    {"id": "rest_request", "text": "{caller} requests a rest halt and gives their intended timing.", "starter": "caller", "min": 3, "max": 5},
    {"id": "relief_timing", "text": "{caller} asks when their relief is due to arrive.", "starter": "caller", "min": 3, "max": 5},
    {"id": "brief_reminder", "text": "{other} reminds {caller} of a briefing time and asks for acknowledgement.", "starter": "other", "min": 3, "max": 5},

    # --- Observation and reporting ---
    {"id": "unknown_vehicle", "text": "{caller} observes unidentified vehicles moving through their sector and reports the details.", "starter": "caller", "min": 4, "max": 8},
    {"id": "civilians_present", "text": "{caller} reports civilian activity in their area of responsibility and asks for guidance.", "starter": "caller", "min": 4, "max": 7},
    {"id": "livestock_movement", "text": "{caller} reports herders moving livestock across their sector.", "starter": "caller", "min": 3, "max": 5},
    {"id": "abandoned_object", "text": "{caller} reports an abandoned item near their position and requests instructions.", "starter": "caller", "min": 4, "max": 7},
    {"id": "aircraft_overhead", "text": "{caller} reports an aircraft passing over their position and asks if it is friendly.", "starter": "caller", "min": 3, "max": 6},
    {"id": "light_observed", "text": "{caller} reports lights seen at a distance during the hours of darkness.", "starter": "caller", "min": 4, "max": 6},
    {"id": "negative_contact", "text": "{caller} confirms they searched an area and found nothing of note.", "starter": "caller", "min": 3, "max": 5},
    {"id": "track_report", "text": "{caller} reports vehicle tracks found at grid {grid} and estimates their age.", "starter": "caller", "min": 4, "max": 6},

    # --- Weather and environment ---
    {"id": "weather_report", "text": "{caller} reports deteriorating weather that is affecting visibility in their sector.", "starter": "caller", "min": 3, "max": 5},
    {"id": "weather_forecast", "text": "{other} passes a weather warning to {caller} and asks them to acknowledge.", "starter": "other", "min": 3, "max": 5},
    {"id": "river_crossing", "text": "{caller} reports a water crossing is impassable and asks for guidance.", "starter": "caller", "min": 4, "max": 6},
    {"id": "visibility_report", "text": "{other} asks {caller} what visibility is like at their location.", "starter": "other", "min": 3, "max": 5},

    # --- Coordination between units ---
    {"id": "boundary_coordination", "text": "{caller} coordinates with {other} about the boundary between their two sectors.", "starter": "caller", "min": 4, "max": 6},
    {"id": "mutual_sighting", "text": "{caller} asks {other} to confirm whether a vehicle seen nearby belongs to them.", "starter": "caller", "min": 3, "max": 6},
    {"id": "pass_information", "text": "{caller} passes {other} information about conditions in an area they have just left.", "starter": "caller", "min": 4, "max": 6},
    {"id": "orders_acknowledgement", "text": "{other} passes instructions to {caller}, who reads them back to confirm.", "starter": "other", "min": 4, "max": 6},
]


# ===========================================================================
# UNIT PAIRS — 14 of them
#
# Roles matter. A logistics element does not talk like a forward patrol,
# and an observation post that sits still all day sounds different from a
# vehicle patrol on the move.
#
# Patrol-to-patrol pairs are important. A net where every transmission
# passes through the command post sounds like a hub, not a network.
# ===========================================================================

UNIT_PAIRS = [
    # Patrols to command
    {"caller": "BRAVO-6",   "other": "ALPHA-1",   "note": "forward patrol reporting to company command"},
    {"caller": "CHARLIE-3", "other": "ALPHA-1",   "note": "patrol reporting to company command"},
    {"caller": "DELTA-2",   "other": "ALPHA-1",   "note": "vehicle patrol reporting to company command"},
    {"caller": "ECHO-4",    "other": "ALPHA-1",   "note": "patrol reporting to company command"},
    {"caller": "GOLF-7",    "other": "ALPHA-1",   "note": "static observation post reporting to company command"},

    # Patrol to patrol
    {"caller": "BRAVO-6",   "other": "CHARLIE-3", "note": "two patrols coordinating directly"},
    {"caller": "DELTA-2",   "other": "ECHO-4",    "note": "two patrols coordinating directly"},
    {"caller": "CHARLIE-3", "other": "GOLF-7",    "note": "patrol coordinating with an observation post"},
    {"caller": "ECHO-4",    "other": "BRAVO-6",   "note": "two patrols coordinating directly"},

    # Logistics and support
    {"caller": "ALPHA-1",   "other": "FOXTROT-5", "note": "command tasking the logistics element"},
    {"caller": "DELTA-2",   "other": "FOXTROT-5", "note": "patrol dealing with logistics directly"},
    {"caller": "FOXTROT-5", "other": "ECHO-4",    "note": "logistics element coordinating a delivery"},

    # Up the chain
    {"caller": "ALPHA-1",   "other": "HOTEL-9",   "note": "company command to battalion watchkeeper"},
    {"caller": "HOTEL-9",   "other": "ALPHA-1",   "note": "battalion watchkeeper calling company command"},
]


# ===========================================================================
# DETAIL SLOTS
#
# Nearly free variety. The model writes differently for "night, fog, weak
# signal, tired operator" than for "midday, clear, strong signal,
# experienced operator" even given the same situation and same units.
#
# Repeated entries are weighted more likely. Most traffic should be
# routine, in clear conditions, with a straightforward shape.
# ===========================================================================

TIMES = [
    "just before dawn", "at first light", "early morning", "mid-morning",
    "around midday", "early afternoon", "mid-afternoon", "late afternoon",
    "at dusk", "just after dark", "late at night", "in the small hours",
]

WEATHER = [
    "clear and still", "clear but very cold", "light rain", "heavy rain",
    "thick fog", "patchy mist", "strong wind", "blowing dust",
    "overcast and grey", "very hot and hazy", "sleet", "bright and glaring",
]

SIGNAL = [
    "signal is strong and clear",
    "signal is strong and clear",
    "signal is workable",
    "signal is weak but readable",
    "signal keeps breaking up",
    "there is interference on the channel",
]

URGENCY = [
    "routine traffic, no hurry",
    "routine traffic, no hurry",
    "routine traffic, no hurry",
    "routine traffic, no hurry",
    "priority traffic, keep it brief",
]

TEXTURE = [
    "the exchange is completely straightforward",
    "the exchange is completely straightforward",
    "the exchange is completely straightforward",
    "one transmission has to be sent again because it was not heard",
    "the receiving station asks a short clarifying question",
    "the receiving station tells the caller to wait, then comes back",
    "the caller reads back the instruction to confirm it",
    "the receiving station is slightly impatient and cuts it short",
    "the caller corrects themselves mid-message",
    "the receiving station acknowledges with the bare minimum of words",
]

OPERATOR = [
    "the operator is experienced and very economical with words",
    "the operator is experienced and very economical with words",
    "the operator is competent but slightly formal",
    "the operator is tired near the end of a long shift",
    "the operator is new and sticks rigidly to correct procedure",
]

# Fixed fictional geography. One small consistent area, so units do not
# teleport around the map across a session. Spoken as digits throughout.
GRIDS = [
    "four two six eight", "four three seven one", "four four six five",
    "four two five nine", "four five seven zero", "four three eight four",
    "four six six two", "four four nine seven", "four five six three",
    "four three five two",
]


# ===========================================================================
# ROTATING EXAMPLES
#
# A single fixed example anchors the model's phrasing hard and is a hidden
# cause of repetition. Rotating through several styles breaks that.
# ===========================================================================

EXAMPLE_STYLES = [
    [("{a}", "{b}, this is {a}, radio check, over."),
     ("{b}", "{a}, {b}, strength five, over.")],

    [("{a}", "{b}, {a}, message follows, over."),
     ("{b}", "{a}, send it, over.")],

    [("{a}", "{b}, this is {a}, at checkpoint two, over."),
     ("{b}", "{a}, roger, out.")],

    [("{a}", "{b}, {a}, request permission to move, over."),
     ("{b}", "{a}, wait, out.")],

    [("{a}", "{b}, this is {a}, say again your last, over."),
     ("{b}", "{a}, I say again, grid four two six eight, over.")],

    [("{a}", "{b}, {a}, sitrep to follow, over."),
     ("{b}", "{a}, {b}, ready to copy, over.")],
]


# ===========================================================================
# EVENT THREADS
#
# Linked exchanges over time, sharing a thread id. The Phase 4 orchestrator
# plays them at the given second offsets with unrelated routine traffic
# filling the gaps, so a listener hears a situation develop rather than
# unrelated fragments.
# ===========================================================================

THREAD_TEMPLATES = [
    {
        "id": "vehicle_sighting_thread",
        "stages": [
            {"text": "{caller} reports unidentified vehicles moving through their sector at grid {grid}.", "gap": 0, "min": 4, "max": 7},
            {"text": "{other} asks {caller} for an update on the vehicles reported earlier.", "gap": 240, "min": 3, "max": 6, "starter": "other"},
            {"text": "{caller} reports the vehicles have left the area and the sector is clear.", "gap": 600, "min": 3, "max": 5},
        ],
    },
    {
        "id": "resupply_thread",
        "stages": [
            {"text": "{caller} requests resupply of water and batteries at grid {grid}.", "gap": 0, "min": 4, "max": 6},
            {"text": "{other} confirms a resupply run is being arranged and gives a rough timing.", "gap": 420, "min": 3, "max": 5, "starter": "other"},
            {"text": "{caller} confirms the resupply has arrived and been received.", "gap": 1500, "min": 3, "max": 4},
        ],
    },
    {
        "id": "vehicle_fault_thread",
        "stages": [
            {"text": "{caller} reports a vehicle breakdown at grid {grid} and requests recovery.", "gap": 0, "min": 4, "max": 6},
            {"text": "{other} tells {caller} a recovery team is on the way and asks them to hold position.", "gap": 300, "min": 3, "max": 5, "starter": "other"},
            {"text": "{caller} reports the recovery team has arrived and work has begun.", "gap": 1200, "min": 3, "max": 5},
            {"text": "{caller} reports the vehicle is running again and they are moving on.", "gap": 2400, "min": 3, "max": 4},
        ],
    },
    {
        "id": "route_block_thread",
        "stages": [
            {"text": "{caller} reports their planned route is blocked at grid {grid}.", "gap": 0, "min": 3, "max": 6},
            {"text": "{other} passes an alternative route to {caller}, who reads it back.", "gap": 180, "min": 4, "max": 6, "starter": "other"},
            {"text": "{caller} confirms they are clear of the obstruction and back on schedule.", "gap": 900, "min": 3, "max": 4},
        ],
    },
    {
        "id": "comms_problem_thread",
        "stages": [
            {"text": "{other} reports they are receiving {caller} very poorly and asks them to move location.", "gap": 0, "min": 3, "max": 6, "starter": "other"},
            {"text": "{caller} has moved to higher ground and calls again to test communications.", "gap": 360, "min": 3, "max": 5},
        ],
    },
    {
        "id": "sick_soldier_thread",
        "stages": [
            {"text": "{caller} reports a member of the patrol is unwell and asks for advice.", "gap": 0, "min": 4, "max": 6},
            {"text": "{other} asks {caller} for more detail on the condition of the person.", "gap": 200, "min": 3, "max": 6, "starter": "other"},
            {"text": "{caller} reports the person has improved and the patrol will continue.", "gap": 1100, "min": 3, "max": 5},
        ],
    },
    {
        "id": "abandoned_object_thread",
        "stages": [
            {"text": "{caller} reports an abandoned item near grid {grid} and requests instructions.", "gap": 0, "min": 4, "max": 6},
            {"text": "{other} instructs {caller} to keep clear and await a specialist team.", "gap": 240, "min": 3, "max": 5, "starter": "other"},
            {"text": "{caller} confirms the specialist team has arrived and taken over.", "gap": 1800, "min": 3, "max": 4},
        ],
    },
    {
        "id": "overdue_patrol_thread",
        "stages": [
            {"text": "{other} notes {caller} is overdue for their scheduled report and calls them.", "gap": 0, "min": 3, "max": 5, "starter": "other"},
            {"text": "{caller} explains they were delayed by ground conditions and gives a new timing.", "gap": 120, "min": 3, "max": 6},
            {"text": "{caller} reports they have arrived and are going firm.", "gap": 1400, "min": 2, "max": 4},
        ],
    },
    {
        "id": "weather_deterioration_thread",
        "stages": [
            {"text": "{caller} reports visibility is dropping quickly in their sector.", "gap": 0, "min": 3, "max": 5},
            {"text": "{other} asks {caller} to report their current visibility.", "gap": 300, "min": 3, "max": 5, "starter": "other"},
            {"text": "{other} instructs {caller} to shorten the patrol and return early.", "gap": 800, "min": 3, "max": 6, "starter": "other"},
            {"text": "{caller} confirms they are back at base and the patrol is closed down.", "gap": 2000, "min": 2, "max": 4},
        ],
    },
]


# ===========================================================================
# RULES
# ===========================================================================

BANNED = [
    "over and out",      # contradiction: "over" wants a reply, "out" ends it
    "repeat,",           # on a real net this means fire the artillery again
    "repeat that",
    "please repeat",
    "i repeat",
    "10-4", "ten-four",  # civilian CB, not military
    "breaker",
    "what's your twenty",
    "do you copy",       # film language
    "come in",
    "roger wilco",       # redundant, wilco already includes roger
    "plate number", "license plate",
    "binos",
    "mayday",
    "loud and proud",
    "godspeed",
    "good luck out there",
]

VALID_ENDINGS = ("over", "out", "wilco")

SYSTEM_MESSAGE = (
    "You write realistic military radio traffic for a training simulator. "
    "You know real voice procedure. Transmissions are brief and functional: "
    "usually one short sentence, ending in a proword. Real radio traffic is "
    "dull and administrative, not dramatic. You never write film dialogue, "
    "never add explanation, and never add a preamble, title, or notes."
)

JUDGE_SYSTEM = (
    "You are an experienced signals instructor reviewing radio traffic "
    "written for a training simulator. You are strict. You reward brevity, "
    "correct voice procedure, and dull realism. You penalise anything that "
    "sounds like a film, anything wordy, and any misuse of prowords."
)


# ===========================================================================
# TALKING TO THE MODEL
# ===========================================================================

def ask(model, system_message, user_message, json_mode=True, max_tokens=500):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "keep_alive": "60m",     # stop Ollama unloading between calls
        "options": {
            "temperature": 0.75,
            "top_p": 0.9,
            "repeat_penalty": 1.12,
            "num_predict": max_tokens,
            "num_ctx": 2048,
        },
    }
    if json_mode:
        payload["format"] = "json"

    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def render_example(starter, second):
    """Pick one rotating example style and fill in the callsigns."""
    style = random.choice(EXAMPLE_STYLES)
    lines = []
    for who, text in style:
        speaker = starter if who == "{a}" else second
        lines.append((speaker, text.format(a=starter, b=second)))

    if USE_JSON:
        items = ",\n  ".join(
            f'{{"callsign": "{s}", "text": "{t}"}}' for s, t in lines)
        return ("Reply with JSON in exactly this shape and nothing else:\n\n"
                f'{{"transmissions": [\n  {items}\n]}}')

    body = "\n".join(f'{s}: "{t}"' for s, t in lines)
    return ("Write one transmission per line, exactly like this, "
            f"nothing else:\n\n{body}")


def build_message(text, pair, details, starter, second, min_n, max_n,
                  avoid=None, context=None):
    parts = [f"SITUATION: {text}", ""]

    if context:
        parts += ["EARLIER TRAFFIC ON THIS EVENT "
                  "(refer back to it naturally, do not repeat it):",
                  context, ""]

    parts += [
        f'UNITS: {pair["caller"]} and {pair["other"]} — {pair["note"]}',
        f"{starter} speaks first.",
        "",
        "CONDITIONS:",
        f'- Time: {details["time"]}',
        f'- Weather: {details["weather"]}',
        f'- Radio conditions: {details["signal"]}',
        f'- Priority: {details["urgency"]}',
        f'- Shape of the exchange: {details["texture"]}',
        f'- Operator: {details["operator"]}',
        "",
        render_example(starter, second),
        "",
        "RULES:",
        f"- Between {min_n} and {max_n} transmissions.",
        f'- Speakers strictly alternate. Only {pair["caller"]} and {pair["other"]} appear.',
        '- Every transmission ends with "over", "wilco", or "out".',
        '- The final transmission ends with "out".',
        "- Usually one short sentence each. Radio traffic is brief.",
        '- Say "say again" if something needs repeating. NEVER say "repeat".',
        '- Never write "over and out".',
        '- Numbers as spoken digits: "four two six eight", not "4268".',
        f'- If a grid is mentioned, use {details["grid"]}.',
        "- Keep it dull and procedural. No drama, no slang, no small talk.",
    ]

    if avoid:
        parts += ["",
                  "AVOID these openings — they are already overused:",
                  "\n".join(f'- "{a}..."' for a in avoid)]

    return "\n".join(parts)


def judge(exchange_lines):
    """Score realism 1-5. Short output, so this is cheap even on a big model."""
    body = "\n".join(f'{c}: "{t}"' for c, t in exchange_lines)

    prompt = f"""Rate this radio exchange for realism.

{body}

Scoring:
5 = indistinguishable from a real radio log
4 = good, minor stiffness
3 = usable but generic or slightly wordy
2 = sounds written, film-like phrasing
1 = clearly wrong procedure or nonsense

Reply with JSON only: {{"score": <1-5>, "reason": "<8 words max>"}}"""

    try:
        raw = ask(JUDGE_MODEL, JUDGE_SYSTEM, prompt, json_mode=True,
                  max_tokens=80)
        data = json.loads(raw)
        return int(data.get("score", 3)), str(data.get("reason", ""))[:60]
    except (requests.RequestException, json.JSONDecodeError, ValueError, TypeError):
        return 3, "judge failed"     # neutral, never blocks the run


# ===========================================================================
# CHECKING THE OUTPUT
# ===========================================================================

def parse_text(raw):
    """Used only when USE_JSON is False."""
    out = []
    pattern = re.compile(r'^([A-Z][A-Z0-9\-]{1,15}):\s*["\']?(.+?)["\']?\s*$')
    for line in raw.strip().splitlines():
        m = pattern.match(line.strip())
        if m:
            out.append({"callsign": m.group(1).strip(),
                        "text": m.group(2).strip()})
    return {"transmissions": out}


def normalise(text):
    """Strip callsigns and numbers so we compare phrasing, not detail."""
    t = text.lower()
    t = re.sub(r'\b(alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel)[- ]?\d*\b', 'X', t)
    t = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|zero|niner)\b', 'N', t)
    t = re.sub(r'\d+', 'N', t)
    t = re.sub(r'[^\w\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def validate(data, pair, min_n, max_n):
    """Return None if good, or a short reason string if bad."""
    if not isinstance(data, dict) or "transmissions" not in data:
        return "no transmissions key"

    tx = data["transmissions"]
    if not isinstance(tx, list):
        return "transmissions not a list"

    n = len(tx)
    if n < min_n or n > max_n:
        return f"length {n} outside {min_n}-{max_n}"

    allowed = {pair["caller"].upper(), pair["other"].upper()}
    previous = None

    for i, item in enumerate(tx):
        if not isinstance(item, dict) or "callsign" not in item or "text" not in item:
            return f"line {i+1} malformed"

        callsign = str(item["callsign"]).strip().upper()
        text = str(item["text"]).strip()
        low = text.lower()

        if callsign not in allowed:
            return f"line {i+1} stray callsign {callsign}"
        if callsign == previous:
            return f"line {i+1} {callsign} transmits twice in a row"
        previous = callsign

        wc = len(text.split())
        if wc > 32:
            return f"line {i+1} too long ({wc} words)"
        if wc < 3:
            return f"line {i+1} too short"

        for bad in BANNED:
            if bad in low:
                return f"line {i+1} banned phrase: {bad}"

        ending = low.rstrip(' .!?"\'')
        if i == n - 1:
            if not ending.endswith("out"):
                return "final line does not end with 'out'"
        elif not ending.endswith(VALID_ENDINGS):
            return f"line {i+1} bad ending: ...{text[-24:]}"

    return None


def too_similar(data, seen, threshold=0.7):
    """Reject an exchange whose lines mostly already exist in the pool."""
    sigs = [normalise(t["text"]) for t in data["transmissions"]]
    if not sigs:
        return True
    return sum(1 for s in sigs if s in seen) / len(sigs) >= threshold


def roll_details():
    return {
        "time": random.choice(TIMES),
        "weather": random.choice(WEATHER),
        "signal": random.choice(SIGNAL),
        "urgency": random.choice(URGENCY),
        "texture": random.choice(TEXTURE),
        "operator": random.choice(OPERATOR),
        "grid": random.choice(GRIDS),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    # Settings from the top of the file, with the command line able to
    # override any of them.
    args = sys.argv[1:]

    target = HOW_MANY
    max_attempts = MAX_ATTEMPTS
    fresh = FRESH_START or "--fresh" in args
    do_judge = ENABLE_JUDGE or "--judge" in args
    thread_mode = (MODE == "threads") or "--threads" in args

    for a in args:
        if a.isdigit():
            target = int(a)

    pool = []
    if not fresh and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            pool = json.load(f)
        print(f"Resuming: {len(pool)} exchanges already saved")

    seen = {normalise(t["text"]) for ex in pool for t in ex["transmissions"]}
    openings = Counter(" ".join(t["text"].split()[:4]).lower()
                       for ex in pool for t in ex["transmissions"])

    def save():
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, ensure_ascii=False)

    def on_interrupt(signum, frame):
        print("\n\nInterrupted. Saving...")
        save()
        print(f"Saved {len(pool)}. Run the script again to continue.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    stats = Counter()
    scores = []
    started = time.time()
    consecutive_fails = 0
    attempts = 0

    print(f"Generator: {GEN_MODEL}")
    print(f"Judge:     {JUDGE_MODEL if do_judge else 'disabled (set ENABLE_JUDGE = True)'}")
    print(f"Mode:      {'event threads' if thread_mode else 'single exchanges'}")
    print(f"Library:   {len(SITUATIONS)} situations x {len(UNIT_PAIRS)} pairs "
          f"= {len(SITUATIONS)*len(UNIT_PAIRS)} base combinations")
    print(f"Target:    {target} exchanges, max {max_attempts} attempts")
    print("Ctrl+C is safe — progress is saved.")
    print()

    def generate_one(text, pair, details, starter, second, min_n, max_n,
                     context=None):
        """Returns (data, reason_if_rejected, score)."""
        avoid = [p for p, c in openings.most_common(6) if c >= 4]
        msg = build_message(text, pair, details, starter, second,
                            min_n, max_n, avoid=avoid, context=context)
        try:
            raw = ask(GEN_MODEL, SYSTEM_MESSAGE, msg, json_mode=USE_JSON)
            data = json.loads(raw) if USE_JSON else parse_text(raw)
        except json.JSONDecodeError:
            return None, "invalid JSON", 0
        except requests.RequestException as e:
            print(f"\nConnection problem: {e}")
            print("Is Ollama running? Try 'ollama serve' in another window.")
            raise SystemExit(1)

        problem = validate(data, pair, min_n, max_n)
        if problem:
            return None, problem.split(":")[0].split("(")[0].strip(), 0
        if too_similar(data, seen):
            return None, "too similar to existing", 0

        score = 5
        if do_judge:
            lines = [(t["callsign"], t["text"]) for t in data["transmissions"]]
            score, reason = judge(lines)
            if score < MIN_SCORE:
                return None, f"judge scored {score}", score

        return data, None, score

    def commit(data, meta):
        for t in data["transmissions"]:
            seen.add(normalise(t["text"]))
            openings[" ".join(t["text"].split()[:4]).lower()] += 1
        pool.append({"id": len(pool) + 1, **meta,
                     "transmissions": data["transmissions"]})

    def progress(label):
        elapsed = time.time() - started
        rate = len(pool) / elapsed if elapsed else 0
        eta = (target - len(pool)) / rate / 60 if rate else 0
        print(f"  {len(pool)}/{target}  {label}  "
              f"[try {attempts}/{max_attempts}]  ETA {eta:.0f} min", flush=True)

    # -----------------------------------------------------------------------
    # THREAD MODE
    # -----------------------------------------------------------------------
    if thread_mode:
        while len(pool) < target and attempts < max_attempts:
            attempts += 1
            template = random.choice(THREAD_TEMPLATES)
            pair = random.choice(UNIT_PAIRS)
            base = roll_details()
            thread_id = f"T{random.randint(100000, 999999)}"
            context_lines = []
            ok = True

            for idx, stage in enumerate(template["stages"]):
                starter = (pair["caller"]
                           if stage.get("starter", "caller") == "caller"
                           else pair["other"])
                second = (pair["other"] if starter == pair["caller"]
                          else pair["caller"])
                text = stage["text"].format(caller=pair["caller"],
                                            other=pair["other"],
                                            grid=base["grid"])

                details = roll_details()
                details["grid"] = base["grid"]     # geography stays consistent

                data, reason, score = generate_one(
                    text, pair, details, starter, second,
                    stage["min"], stage["max"],
                    context="\n".join(context_lines) if context_lines else None)

                if data is None:
                    stats[reason] += 1
                    ok = False
                    break

                commit(data, {
                    "scenario": template["id"],
                    "thread_id": thread_id,
                    "thread_stage": idx,
                    "thread_gap_seconds": stage["gap"],
                    "units": [pair["caller"], pair["other"]],
                    "conditions": details,
                    "score": score,
                })
                scores.append(score)
                context_lines += [f'{t["callsign"]}: "{t["text"]}"'
                                  for t in data["transmissions"]]

            if ok:
                consecutive_fails = 0
                progress(f"thread {template['id']:28s}")
            else:
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    print("\nToo many consecutive failures. Stopping.")
                    print("Most common:", stats.most_common(3))
                    break

            save()

    # -----------------------------------------------------------------------
    # SINGLE EXCHANGE MODE
    # -----------------------------------------------------------------------
    else:
        combos = [(s, p) for s in SITUATIONS for p in UNIT_PAIRS]
        random.shuffle(combos)
        i = 0

        while len(pool) < target and attempts < max_attempts:
            attempts += 1

            if i >= len(combos):        # completed a full pass, reshuffle
                random.shuffle(combos)
                i = 0
            situation, pair = combos[i]
            i += 1

            details = roll_details()
            starter = (pair["caller"] if situation["starter"] == "caller"
                       else pair["other"])
            second = pair["other"] if starter == pair["caller"] else pair["caller"]
            text = situation["text"].format(caller=pair["caller"],
                                            other=pair["other"],
                                            grid=details["grid"])

            data, reason, score = generate_one(
                text, pair, details, starter, second,
                situation["min"], situation["max"])

            if data is None:
                stats[reason] += 1
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    print(f"\n{MAX_CONSECUTIVE_FAILS} failures in a row. Stopping.")
                    print("Most common:", stats.most_common(3))
                    break
                continue

            consecutive_fails = 0
            commit(data, {
                "scenario": situation["id"],
                "units": [pair["caller"], pair["other"]],
                "conditions": details,
                "score": score,
            })
            scores.append(score)

            tag = f"score {score}" if do_judge else ""
            progress(f"[{situation['id']:22s} {pair['caller']:>9s}] {tag}")

            if len(pool) % 25 == 0:
                save()

    save()

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    elapsed = time.time() - started
    tried = len(pool) + sum(stats.values())

    print()
    print("=" * 62)
    print(f"Pool: {len(pool)} exchanges in {elapsed/60:.0f} minutes")
    print(f"Attempts used: {attempts} of {max_attempts}")
    if tried:
        print(f"Keep rate: {len(pool)/tried*100:.0f}%")

    if attempts >= max_attempts and len(pool) < target:
        print()
        print("Hit the attempt limit before reaching the target.")
        print("Either raise MAX_ATTEMPTS, or look at the rejection reasons")
        print("below — a dominant reason usually means a rule is too strict.")
    if do_judge and scores:
        print(f"Average realism score: {sum(scores)/len(scores):.2f} / 5")

    if stats:
        print()
        print("Rejections:")
        for reason, count in stats.most_common(10):
            print(f"   {count:4d}  {reason}")

    lines = [t["text"] for ex in pool for t in ex["transmissions"]]
    if lines:
        unique = len({normalise(t) for t in lines})
        variety = unique / len(lines) * 100
        print()
        print(f"Transmissions: {len(lines)}")
        print(f"Unique phrasings: {unique}  ({variety:.0f}% variety)")
        if variety < 60:
            print("  LOW. Something is wrong — check the rejection reasons.")
        elif variety < 80:
            print("  Acceptable, with room to improve.")
        else:
            print("  Good.")

        print()
        print("Top scenarios:")
        for name, count in Counter(ex["scenario"] for ex in pool).most_common(8):
            print(f"   {count:4d}  {name}")

    print()
    print(f"Saved to {OUTPUT_FILE}")
    print("Next: python step2_curate.py stats")


if __name__ == "__main__":
    main()
